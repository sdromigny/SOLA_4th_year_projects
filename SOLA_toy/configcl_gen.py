import configparser

class Cfig(object):
    def __init__(self, fileini):
        cfg = configparser.ConfigParser()
        cfg.read(fileini)

        # read toy problem parameters if section 'toy_problem exist'
        #------------------------------------------
        # only for toy problem
        if cfg.has_section('toy_problem'):
            self.nx = cfg.getint('toy_problem', 'nx')
            self.ny = cfg.getint('toy_problem', 'ny')
            self.factmul = cfg.getfloat('toy_problem', 'factmul')

        # general parameters
        #------------------------------------------
        # for all problems
        self.flag_new = cfg.getboolean('gen_params', 'flag_new')

        self.radname = cfg.get('gen_params', 'radname')

        # preprocessing
        #------------------------------------------
        self.flag_sigma = cfg.getboolean('preprocessing', 'flag_sigma')
        self.flag_preprocess = cfg.getboolean('preprocessing', 'flag_preprocess')
        
        # paths 
        #------------------------------------------
        self.indir = cfg.get('paths', 'indir')
        self.outdir = cfg.get('paths', 'outdir')

        # SOLA inversion parameters
        #------------------------------------------
        self.SOLAlsqr = cfg.get('inversion','SOLAlsqr')
        self.niter = cfg.getint('inversion','niter')
        self.eta = cfg.getfloat('inversion','eta')
        self.kmin = cfg.getint('inversion','kmin')
        self.kmax = cfg.getint('inversion','kmax')

        # Filtering
        #------------------------------------------
        if cfg.has_section('filtering'):
            self.input_model = cfg.get('filtering','input_model')

        # postprocessing 
        #------------------------------------------
        self.compAkC = cfg.get('postprocessing','compAkC')
        self.do_Ak = cfg.getboolean('postprocessing', 'do_Ak')
        self.do_mk = cfg.getboolean('postprocessing', 'do_mk')
        self.do_Rmatrix = cfg.getboolean('postprocessing', 'do_Rmatrix')

        # Plotting parameters
        #------------------------------------------
        if cfg.has_section('Plotting'):
            self.outdirfig = cfg.get('Plotting','outdirfig')
            self.do_relative = cfg.getboolean('Plotting', 'do_relative')
        
            self.vmin = cfg.getfloat('Plotting','vmin')
            self.vmax = cfg.getfloat('Plotting','vmax')
            self.sigmax = cfg.getfloat('Plotting','sigmax')

            ## for geographic plots
            #self.central_longitude = cfg.getfloat('Plotting','central_longitude')
            #self.central_latitude = cfg.getfloat('Plotting','central_latitude')
            #self.latmin = cfg.getfloat('Plotting','latmin')
            #self.latmax = cfg.getfloat('Plotting','latmax')
            #self.lonmin = cfg.getfloat('Plotting','lonmin')
            #self.lonmax = cfg.getfloat('Plotting','lonmax')

            # Averaging kernel Plotting parameters
            #------------------------------------------
            self.kvalue = cfg.getint('PlottingKernel','kvalue')


    def getlistint(self, string):
        elts = string.split(',')
        listint = []
        for elt in elts:
            listint.append(int(elt))

        return listint

    def getlistfloat(self, string):
        elts = string.split(',')
        listfloat = []
        for elt in elts:
            listfloat.append(float(elt))

        return listfloat

    def getlist(self, string):
        elts = string.split(',')
        liststr = []
        for elt in elts:
            liststr.append(elt)

        return liststr

