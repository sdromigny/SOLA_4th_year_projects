import numpy as np
import os
import numpy as np
import pandas as pd
import csv
import pykonal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from tqdm import tqdm  # optional progress bar; install with `pip install tqdm`


# ---- Compatibility patch for NumPy >= 2.0 (pykonal uses np.infty in some versions) ----
if not hasattr(np, "infty"):
    np.infty = np.inf

# -------------------- Parsing utilities --------------------
# def parse_dat_file(dat_path):
#     """
#     Parse the .dat file format you showed.
#     Returns: dict mapping source_id -> list of records
#     each record: {'station': str, 'tt': float, 'weight': float, 'phase': 'P'/'S'}
#     """
#     sources = {}
#     current_source = None
#     with open(dat_path, "r") as f:
#         for raw in f:
#             line = raw.strip()
#             if not line:
#                 continue
#             if line.startswith("#"):
#                 # header line; take last token as source id (strip '#')
#                 tokens = line[1:].strip().split()
#                 if len(tokens) == 0:
#                     continue
#                 current_source = tokens[-1]
#                 if current_source not in sources:
#                     sources[current_source] = []
#                 continue

#             # parse entry lines like: NAME   0.70   1.00  P
#             parts = line.split()
#             if len(parts) < 4:
#                 # skip malformed lines
#                 continue
#             station = parts[0]
#             try:
#                 tt = float(parts[1])
#             except Exception:
#                 tt = np.nan
#             try:
#                 weight = float(parts[2])
#             except Exception:
#                 weight = np.nan
#             phase = parts[3].upper()
#             if current_source is None:
#                 # if file starts without header, skip
#                 continue
#             sources[current_source].append({"station": station, "tt": tt, "weight": weight, "phase": phase})
#     return sources

# -------------------- Station/source coordinate loaders --------------------
def load_station_coords_csv(station_csv):
    """
    station_csv: CSV with header: name,x,y,z  (z optional; if missing, z=0)
    Returns dict: name -> np.array([x,y,z])
    """
    coords = {}
    with open(station_csv, newline="") as csvfile:
        rdr = csv.DictReader(csvfile)
        for row in rdr:
            name = row.get("name") or row.get("station") or row.get("sta") or row.get("s")
            if name is None:
                continue
            x = float(row.get("x_m"))/1000
            y = float(row.get("y_m"))/1000
            z = float(row.get("elev"))/1000
            coords[name] = np.array([x, y, z], dtype=float)
    return coords

def load_source_coords_csv(source_csv):
    """
    Read events / sources CSV and return dict mapping source_id (string) -> np.array([x_km, y_km, z_km]).
    Handles common column names: 'name','id','event','event_id'. Accepts whitespace-delimited files.
    Converts x_m,y_m,elev from meters -> kilometers (divide by 1000).
    """
    import pandas as pd
    coords = {}

    # read with whitespace delimiting (your sample file looks whitespace-separated)
    df = pd.read_csv(source_csv, delim_whitespace=True, dtype=str)
    # strip whitespace from column names
    df.columns = df.columns.str.strip()

    # normalize column names to lower-case for detection
    cols_lower = {c.lower(): c for c in df.columns}

    # try to find id column
    id_col = None
    for candidate in ("name", "id", "event", "event_id", "name_id"):
        if candidate in cols_lower:
            id_col = cols_lower[candidate]
            break
    # fallback: if 'date'+'time' pair could be used as unique id, create one
    if id_col is None:
        # if there's a 'name' like column with numeric labels, pandas may have parsed it differently.
        # fall back to using the row index as ID (string)
        df["__generated_id__"] = df.index.astype(str)
        id_col = "__generated_id__"

    # try to find coordinate columns
    x_col = cols_lower.get("x_m") or cols_lower.get("x") or next((c for c in df.columns if c.lower().startswith("x")), None)
    y_col = cols_lower.get("y_m") or cols_lower.get("y") or next((c for c in df.columns if c.lower().startswith("y")), None)
    z_col = cols_lower.get("elev") or cols_lower.get("z") or cols_lower.get("depth") or next((c for c in df.columns if c.lower().startswith("e")), None)

    if x_col is None or y_col is None:
        raise ValueError(f"Could not detect x/y columns in {source_csv}. Found columns: {list(df.columns)}")

    # coerce numeric columns (safe conversion), fill or raise if not convertible
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    if z_col is not None:
        df[z_col] = pd.to_numeric(df[z_col], errors="coerce")
    else:
        # if no z/elev, create zeros
        df["__z__"] = 0.0
        z_col = "__z__"

    # convert meters -> kilometers consistently
    df[x_col] = df[x_col] / 1000.0
    df[y_col] = df[y_col] / 1000.0
    df[z_col] = df[z_col]

    # build dict mapping string ids to np.array coords
    for _, row in df.iterrows():
        raw_id = row[id_col]
        if pd.isna(raw_id):
            continue
        src_id = str(raw_id).strip()
        if pd.isna(row[x_col]) or pd.isna(row[y_col]) or pd.isna(row[z_col]):
            # skip or warn if coordinate missing
            print(f"[WARN] skipping source {src_id} because of missing coordinates")
            continue
        coords[src_id] = np.array([row[x_col], row[y_col], row[z_col]], dtype=float)

    # helpful debug print
    print(f"[INFO] Loaded {len(coords)} source coords from {source_csv} (id column: '{id_col}', x/y/z: '{x_col}/{y_col}/{z_col}')")
    # optionally show a few keys
    sample_keys = list(coords.keys())[:8]
    print(f"[INFO] example source IDs: {sample_keys}")
    return coords


def point_inside_domain(pt, solver):
    minc = np.array(solver.velocity.min_coords, dtype=float)
    intervals = np.array(solver.velocity.node_intervals, dtype=float)
    npts = np.array(solver.velocity.npts, dtype=int)
    maxc = minc + (npts - 1) * intervals
    return np.all(pt >= minc - 1e-8) and np.all(pt <= maxc + 1e-8)


def read_G_text(filename, assume_one_based=True):
    """
    Read the text format written by write_G_text. Returns csr_matrix G.
    The reader will attempt to detect 1-based vs 0-based: if the max index equals N or M,
    it will subtract 1 (i.e., interpret as 1-based). If unsure, use assume_one_based flag.
    """
    with open(filename, "r") as f:
        first = f.readline().strip()
        if not first:
            raise ValueError("Empty file")
        nnz = int(first)
        second = f.readline().strip()
        N_str, M_str = second.split()
        N = int(N_str); M = int(M_str)
        rows = []
        cols = []
        vals = []
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            i = int(parts[0])
            j = int(parts[1])
            v = float(parts[2])
            rows.append(i)
            cols.append(j)
            vals.append(v)

    # detect one-based if requested/detected
    rows = np.array(rows, dtype=int)
    cols = np.array(cols, dtype=int)
    if len(rows) == 0:
        return sp.csr_matrix((N, M), dtype=float)

    # If values appear to be 1-based (max index == N or M) then convert to 0-based
    if assume_one_based:
        rows0 = rows - 1
        cols0 = cols - 1
    else:
        # attempt to auto-detect: if max(rows) == N then probably 1-based
        if rows.max() == N or cols.max() == M:
            rows0 = rows - 1
            cols0 = cols - 1
        else:
            rows0 = rows
            cols0 = cols

    G = sp.coo_matrix((vals, (rows0, cols0)), shape=(N, M)).tocsr()
    return G

def trace_phase_rays(sources_dict, station_coords, source_coords,
                     phase="P", out_dir="rays_output",
                     grid_params=None):
    """
    sources_dict: output of parse_dat_file
    station_coords: dict station->coord
    source_coords: dict source_id->coord
    phase: 'P' or 'S'
    grid_params: dict with keys min_coords,node_intervals,npts,top_layer_vel,other_vel
    """
    os.makedirs(out_dir, exist_ok=True)

    # Defaults if not provided
    if grid_params is None:
        grid_params = {
            "min_coords": (-5,-4,0.0),
            "node_intervals": (1.0,1.0,1.0),
            "npts": (10,10,10),
            "top_layer_vel": 5.5,
            "other_vel": 5.76,
            "coord_sys": "cartesian",
        }

    # prototype solver for domain checks
    proto = build_solver_grid(min_coords=grid_params["min_coords"],
                              node_intervals=grid_params["node_intervals"],
                              npts=grid_params["npts"],
                              top_layer_vel=grid_params["top_layer_vel"],
                              other_vel=grid_params["other_vel"],
                              coord_sys=grid_params.get("coord_sys","cartesian"))
    min_coords = np.array(proto.velocity.min_coords)
    intervals = np.array(proto.velocity.node_intervals)
    npts = proto.velocity.npts

    rays_saved = []
    failures = []

    for src_id, picks in sources_dict.items():
        # filter picks by phase
        picks_phase = [p for p in picks if p["phase"] == phase]
        if len(picks_phase) == 0:
            continue

        # look up source coordinate
        if src_id not in source_coords:
            failures.append(("no_source_coord", src_id, f"Source {src_id} not in source_coords"))
            print(f"[WARN] source {src_id} has no coordinate mapping; skipping")
            continue
        src_coord = np.asarray(source_coords[src_id], dtype=float)
        if not point_inside_domain(src_coord, proto):
            failures.append(("source_outside", src_id, f"Source {src_id} outside domain"))
            print(f"[WARN] source {src_id} at {src_coord} outside solver domain; skipping")
            continue

        # build fresh solver for this source
        solver = build_solver_grid(min_coords=grid_params["min_coords"],
                              node_intervals=grid_params["node_intervals"],
                              npts=grid_params["npts"],
                              top_layer_vel=grid_params["top_layer_vel"],
                              other_vel=grid_params["other_vel"],
                              coord_sys=grid_params.get("coord_sys","cartesian"))

        # locate nearest grid node index for source
        rel = (src_coord - min_coords) / intervals
        src_idx = tuple(np.round(rel).astype(int).tolist())
        src_idx = (max(0, min(npts[0]-1, src_idx[0])),
                   max(0, min(npts[1]-1, src_idx[1])),
                   max(0, min(npts[2]-1, src_idx[2])))
        try:
            solver.traveltime.values[src_idx] = 0.0
        except Exception as e:
            failures.append(("set_tt_fail", src_id, str(e)))
            print(f"[ERROR] Failed to set traveltime at {src_idx} for source {src_id}: {e}")
            continue

        # try to push the source to trial (some PyKonal builds require it)
        try:
            solver.unknown[src_idx] = False
            solver.trial.push(*src_idx)
        except Exception:
            try:
                solver.traveltime.known[src_idx] = True
                solver.traveltime.unknown[src_idx] = False
                try:
                    solver.traveltime.trial.push(*src_idx)
                except Exception:
                    pass
            except Exception:
                # ignore and proceed; many versions are fine
                pass

        # solve eikonal
        try:
            solver.solve()
        except Exception as e:
            failures.append(("solve_fail", src_id, str(e)))
            print(f"[ERROR] solver.solve failed for source {src_id}: {e}")
            continue

        # now trace rays from receivers for this phase
        for pick in picks_phase:
            sta = pick["station"]
            if sta not in station_coords:
                failures.append(("no_sta_coord", src_id, sta))
                print(f"[WARN] station {sta} missing coords; skipping pick for source {src_id}")
                continue
            rec_coord = np.asarray(station_coords[sta], dtype=float)
            if not point_inside_domain(rec_coord, proto):
                failures.append(("rec_outside", src_id, sta))
                print(f"[WARN] receiver {sta} for source {src_id} outside domain; skipping")
                continue

            # trace: backtrace from receiver physical coordinate to the source
            try:
                ray = solver.trace_ray(rec_coord)
                if ray is None or len(ray) == 0:
                    raise RuntimeError("trace_ray returned empty")
            except Exception as e:
                failures.append(("trace_fail", src_id, sta, str(e)))
                print(f"[ERROR] trace_ray failed for src {src_id} -> rec {sta}: {e}")
                continue

            # save and record
            fname = os.path.join(out_dir, f"ray_src{src_id}_rec{sta}_{phase}.npy")
            np.save(fname, ray)
            rays_saved.append({"src": src_id, "rec": sta, "phase": phase, "file": fname, "ray": ray})
            # print(f"Saved ray src={src_id} rec={sta} phase={phase} -> {fname} (path length {len(ray)})")

    return rays_saved, failures

def build_s_ref_from_solver(solver):
    """
    Build reference slowness-per-cell vector s_ref (s/km) from a Pykonal solver
    with a 3D velocity grid.
    
    Assumes:
      - solver.velocity.values shape = (nx, ny, nz)
      - G uses cells indexed as:
            j = ix + iy*ncx + iz*ncx*ncy
        with ncx = nx-1, etc (same as your build_G_from_rays).
    """
    vel_nodes = np.asarray(solver.velocity.values, dtype=float)  # km/s
    nx, ny, nz = vel_nodes.shape
    ncx, ncy, ncz = nx - 1, ny - 1, nz - 1
    M = ncx * ncy * ncz

    s_ref = np.empty(M, dtype=float)

    for iz in range(ncz):
        for iy in range(ncy):
            for ix in range(ncx):
                j = ix + iy * ncx + iz * (ncx * ncy)

                # 8 node velocities around the cell (ix,iy,iz)
                block = vel_nodes[ix:ix+2, iy:iy+2, iz:iz+2]
                v_cell = np.mean(block)   # km/s

                if v_cell <= 0 or np.isnan(v_cell):
                    raise ValueError(f"Non-positive or NaN velocity in cell {j} at ({ix},{iy},{iz})")

                s_ref[j] = 1.0 / v_cell   # s/km

    return s_ref


def build_tobs_and_sigma_from_rays(p_rays, sources_dict):
    """
    Build t_obs and sigma arrays in the same order as p_rays.
    Assumes sources_dict[src_id] is a list of picks with keys:
      'station', 'tt', 'phase', 'unc' (or whatever you named uncertainty)
    """
    t_obs = []
    sigma = []

    # Build a quick index: (src, station, phase) -> (tt, unc)
    # (phase is 'P' or 'S')
    pick_index = {}
    for src_id, picks in sources_dict.items():
        for p in picks:
            key = (src_id, p["station"], p["phase"])
            # if multiple picks exist, you might want to choose best one here
            pick_index[key] = (p["tt"], p.get("unc", p.get("sigma", 0.05)))

    for rec in p_rays:
        key = (rec["src"], rec["rec"], rec["phase"])
        if key not in pick_index:
            raise KeyError(f"No observed travel time for {key}")
        tt, unc = pick_index[key]
        t_obs.append(tt)
        sigma.append(unc)

    return np.array(t_obs, dtype=float), np.array(sigma, dtype=float)



def build_solver_grid(min_coords=(0.0,0.0,0.0), node_intervals=(1.0,1.0,1.0), npts=(10,10,10),
                      top_layer_vel=5.5, other_vel=5.76, coord_sys="cartesian"):
    solver = pykonal.EikonalSolver(coord_sys=coord_sys)
    solver.velocity.min_coords = tuple(min_coords)
    solver.velocity.node_intervals = tuple(node_intervals)
    solver.velocity.npts = tuple(npts)
    nx, ny, nz = solver.velocity.npts
    vel = np.empty((nx, ny, nz), dtype=float)
    vel[:,:,:] = other_vel
    vel[:, :, 0] = top_layer_vel
    solver.velocity.values = vel
    return solver

import pandas as pd

import pandas as pd

def load_event_id_map(events_csv):
    """
    Build mapping (date_str, time_key) -> event_id ('name' column),
    where:
      date_str = 'YYYYMMDD'
      time_key = 'HHMMSS00' (8-digit zero-padded string)

    events_cartesian.csv rows like:
    20221127  6132600  -134.020  233.324  4.007  ...  1  0

    The 'time' column is an integer HHMMSS00 without leading zero on hour.
    """
    df = pd.read_csv(events_csv, sep=r"\s+", dtype=str)

    required_cols = {"date", "time", "name"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{events_csv} must contain columns {required_cols}, got {df.columns}")

    event_id_map = {}

    for _, row in df.iterrows():
        date_str = row["date"].strip()     # e.g. "20221127"
        time_raw = row["time"].strip()     # e.g. "6132600" or "12475300"
        ev_id    = row["name"].strip()     # e.g. "1", "2", ...

        # Interpret time_raw as integer and zero-pad to 8 digits: HHMMSS00
        try:
            tval = int(time_raw)
        except ValueError:
            continue

        time_key = f"{tval:08d}"           # e.g. 6132600 -> "06132600"

        event_id_map[(date_str, time_key)] = ev_id

    # Debug: show a few keys
    print("[DEBUG] event_id_map example keys:")
    for k in list(event_id_map.keys())[:5]:
        print("  ", k, "->", event_id_map[k])

    return event_id_map



import numpy as np

def parse_dat_file(dat_path, event_id_map):
    """
    Parse a phase_error.dat-style file and map each event header to the correct
    event ID ('name' column) from events_cartesian.csv using date+time.

    dat file example:

    # 2022 11 28 21 28 42.00 50.374224 -4.744086 3.748047 0.0 0.0036 0.0032 0.0627 620000
    UNKNOWN
    N105 0.6148 1 P 0.0236
    N101 0.6392 1 P 0.0243
    R137T 1.1986 1 S 0.0614
    ...

    Returns
    -------
    sources : dict
        { event_id (str from 'name' in events_cartesian.csv): [
              {
                'station': str,
                'tt': float,      # measured travel time
                'phase': str,     # 'P' or 'S'
                'unc': float,     # travel-time uncertainty
                'weight': float,  # the 3rd column (often 1)
              }, ...
          ]
        }
    """
    sources = {}
    current_source = None

    with open(dat_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # --- New event header line ---
            if line.startswith("#"):
                tokens = line[1:].strip().split()
                if len(tokens) < 6:
                    current_source = None
                    continue

                # tokens[0:6] = [year, month, day, hour, minute, second]
                year  = int(tokens[0])
                month = int(tokens[1])
                day   = int(tokens[2])
                hour  = int(tokens[3])
                minute= int(tokens[4])
                sec   = float(tokens[5])

                # Build date string: YYYYMMDD (matches events_cartesian)
                date_str = f"{year:04d}{month:02d}{day:02d}"

                # Build time string: HHMMSS00 (matches 12305700 style)
                sec_int = int(sec)  # ignore fractional seconds for now
                time_str = f"{hour:02d}{minute:02d}{sec_int:02d}00"

                key = (date_str, time_str)
                if key not in event_id_map:
                    print(f"[WARN] No event ID found in events_cartesian for date/time {key}, skipping this event")
                    current_source = None
                    continue

                event_id = str(event_id_map[key])  # the 'name' column
                current_source = event_id
                if current_source not in sources:
                    sources[current_source] = []
                continue

            # --- Non-header lines: picks for the current event ---
            if current_source is None:
                # no valid header matched, so ignore
                continue

            parts = line.split()
            # skip junk like 'UNKNOWN'
            if len(parts) < 4:
                continue

            station = parts[0]

            # measured travel time
            try:
                tt = float(parts[1])
            except Exception:
                # malformed -> skip
                continue

            # weight (3rd column)
            try:
                weight = float(parts[2])
            except Exception:
                weight = 1.0

            phase = parts[3].upper()  # 'P' or 'S'

            # optional uncertainty (5th column)
            unc = None
            if len(parts) >= 5:
                try:
                    unc = float(parts[4])
                except Exception:
                    unc = None
            if unc is None:
                unc = 0.05  # fallback default

            sources[current_source].append(
                {
                    "station": station,
                    "tt": tt,
                    "phase": phase,
                    "unc": unc,
                    "weight": weight,
                }
            )

    return sources


if __name__ == "__main__":
    # paths - change to your files
    dat_file = "/workspaces/SOLA_4th_year_projects/DATA_Cornwall/phase_error.dat"
    station_csv = "/workspaces/SOLA_4th_year_projects/DATA_Cornwall/stations_cartesian.csv"       # CSV with columns: name,x,y,z
    source_csv = "/workspaces/SOLA_4th_year_projects/DATA_Cornwall/events_cartesian.csv"         # CSV mapping source-id -> x,y,z
    G_file="/workspaces/SOLA_4th_year_projects/G_P.txt"
    
    G=read_G_text(G_file)


    grid_params = {
            "min_coords": (-5,-4,0.0),
            "node_intervals": (1.0,1.0,1.0),
            "npts": (10,10,10),
            "top_layer_vel": 5.5,
            "other_vel": 5.76,
            "coord_sys": "cartesian",
        }

    # prototype solver for domain checks
    proto = build_solver_grid(min_coords=grid_params["min_coords"],
                              node_intervals=grid_params["node_intervals"],
                              npts=grid_params["npts"],
                              top_layer_vel=grid_params["top_layer_vel"],
                              other_vel=grid_params["other_vel"],
                              coord_sys=grid_params.get("coord_sys","cartesian"))

        # load coords
    station_coords = load_station_coords_csv(station_csv)
    source_coords = load_source_coords_csv(source_csv)
    print(f"Loaded {len(station_coords)} station coords and {len(source_coords)} source coords")

    event_id_map  = load_event_id_map(source_csv)
    sources_dict  = parse_dat_file(dat_file, event_id_map)

    p_rays, p_fail = trace_phase_rays(sources_dict, station_coords, source_coords,
                                  phase="P", out_dir="rays_P", grid_params=grid_params)

    s_ref = build_s_ref_from_solver(proto)
    t_obs, sigma = build_tobs_and_sigma_from_rays(p_rays, sources_dict)

    t_ref = G @ s_ref   # seconds
    d = t_obs - t_ref


    np.savetxt("d", d)
    np.savetxt("dstd", sigma)