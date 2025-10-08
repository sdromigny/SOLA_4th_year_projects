#include "matrice.h"

/*********************************/

/* create a new matrix with zero */

/*********************************/
struct matrix_t *new_matrix(int nb_line, int nb_col)
{
    struct matrix_t *c;
    long int i;

    c = (struct matrix_t *) malloc(sizeof(struct matrix_t));
    assert(c);

    c->nb_line = nb_line;
    c->nb_col = nb_col;
    c->mat = (double **) malloc(c->nb_line * sizeof(double *));

    for (i = 0; i < c->nb_line; i++) {
        (c->mat)[i] = (double *) calloc(c->nb_col, sizeof(double));
        assert((c->mat)[i]);
    }

    return (c);
}

/***************/

/* free matrix */

/***************/
void free_matrix(struct matrix_t *c)
{
    long int i;
    for (i = 0; i < c->nb_line; i++) {
        free((c->mat)[i]);
    }
    free(c->mat);
    free(c);
    c = NULL;
}

void dump_matrix(char *txt, struct matrix_t *m)
{
    long int i, j;

    if (m == NULL) {
        fprintf(stderr, "%s : matrix is not defined\n", txt);
        return;
    }
    fprintf(stderr, "matrix %s : addr=%p  (%ld x %ld)\n", txt, m,
            m->nb_line, m->nb_col);
    for (i = 0; i < m->nb_line; i++) {
        fprintf(stderr, "\t");
        for (j = 0; j < m->nb_col; j++) {
            fprintf(stderr, "%.2f ", (m->mat)[i][j]);
        }
        fprintf(stderr, "\n");
    }
}

/*********************************/

/* create a new vector with zero */

/*********************************/
struct vector_t *new_vector(long int l)
{
    struct vector_t *v;

    v = (struct vector_t *) malloc(sizeof(struct vector_t));
    assert(v);
    v->mat = (double *) calloc(l, sizeof(double));
    assert(v->mat);
    v->length = l;
    return (v);
}

void free_vector(struct vector_t *v)
{
    free(v->mat);
    free(v);
    v = NULL;
}

//________________________________________
// remise a 0 du vecteur  : ajoutee le 11/01/2016
void raz_vector(struct vector_t *v)
{
    long int i;

     for (i = 0; i < v->length; i++)
	v->mat[i] = 0.;
}

//________________________________________
//
void dump_vector(char *s, struct vector_t *v)
{
    long int i;

    fprintf(stderr, "vector %s : ", s);
    for (i = 0; i < v->length; i++)
        fprintf(stderr, "%f ", v->mat[i]);
    fprintf(stderr, "\n");
}

/********************************************************/

/* Chargement de matrice et vecteur a partir de fichier */

/********************************************************/
struct matrix_t *read_matrix(char *filename)
{
    struct matrix_t *a;
    int m, n, i, j;
    FILE *fd;
    int nb_read;

    fprintf(stdout, "reading matrix A from file '%s'\n", filename);
    if (!(fd = fopen(filename, "r"))) {
        perror(filename);
        exit(1);
    }

    nb_read = fscanf(fd, "%d", &m);
    if (nb_read != 1) {
        fprintf(stderr, "Error reading 'm' in '%s'\n", filename);
        exit(1);
    }

    nb_read = fscanf(fd, "%d", &n);
    if (nb_read != 1) {
        fprintf(stderr, "Error reading 'n' in '%s'\n", filename);
        exit(1);
    }

    a = new_matrix(m, n);

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            nb_read = fscanf(fd, "%lf", &(a->mat[i][j]));
            if (nb_read != 1) {
                fprintf(stderr, "Error reading mat[%d][%d] in '%s'\n",
                        i, j, filename);
                exit(1);
            }
        }
    }

    fclose(fd);

    return (a);
}

struct vector_t *import_vector(struct vector_t *b, char *filename)
{
    long int n, j;
    FILE *fd;
    int nb_read;
    long int rayid;
    double val;

    if (!b) {
        b = read_vector(filename);
        return (b);
    }

    fprintf(stdout, "importing vector b from '%s' into (%p) ... ",
            filename, b);
    if (!(fd = fopen(filename, "r"))) {
        perror(filename);
        exit(1);
    }

    nb_read = fscanf(fd, "%ld", &n);
    if (nb_read != 1) {
        fprintf(stderr, "Error reading 'n' in '%s'\n", filename);
        exit(1);
    }

    if (n != b->length) {
        fprintf(stderr,
                "import_vector failed: size required %ld, read %ld in %s\n",
                b->length, n, filename);
        exit(1);
    }

    /* load vector data from file */
    j = 0;
    while (!feof(fd)) {
        nb_read = fscanf(fd, "%ld %lf\n", &rayid, &val);

        if (nb_read != 2) {
            fprintf(stderr, "Error reading mat[%ld] in '%s'\n",
                    j, filename);
            exit(1);
        }

        if (fabs(b->mat[rayid]) > 1.0e-6) {
            fprintf(stdout,
                    "import_vector: duplicate value (%ld) old=%f/new=%f\n",
                    rayid, b->mat[rayid], val);
        }
        b->mat[rayid] = val;
        j++;
    }
    fclose(fd);
    fprintf(stdout, "%ld lines\n", j);
    fflush(stdout);
    return (b);

}

/** \brief read a file as vector
 * 
 * the file is formated as follow :
 *
 * nbitem
 * value1
 * value2
 * ...
 */
struct vector_t *read_simple_vector(char *filename)
{
    struct vector_t *b;
    long int n, j;
    FILE *fd;
    int nb_read;
    double val;

    fprintf(stdout, "reading 'simple' vector b from '%s' ... ", filename);
    if (!(fd = fopen(filename, "r"))) {
        perror(filename);
        exit(1);
    }

    nb_read = fscanf(fd, "%ld", &n);
    if (nb_read != 1) {
        fprintf(stderr, "Error reading 'n' in '%s'\n", filename);
        exit(1);
    }
    fprintf(stderr, "n=%ld\n", n);

    b = new_vector(n);

    /* load vector data from file */
    j = 0;
    while (!feof(fd)) {
        nb_read = fscanf(fd, "%lf\n", &val);
        if (nb_read != 1) {
            fprintf(stderr, "Error reading mat[%ld] in '%s'\n",
                    j, filename);
            exit(1);
        }
        b->mat[j] = val;
        j++;
    }

    if (j != n) {
        fprintf(stderr,
                "read_simple_vector: truncated file ! read only %ld/%ld items.\n",
                j, n);
        exit(1);
    }

    fclose(fd);
    fprintf(stdout, "%ld lines\n", j);
    fflush(stdout);
    return (b);
}

/** \brief read a file as vector
 * 
 * the file is formated as follow :
 *
 * nb_total_of_item_in_vector
 * index1 value1
 * index2 value2
 * ...
 */
struct vector_t *read_vector(char *filename)
{
    struct vector_t *b;
    long int n, j;
    FILE *fd;
    int nb_read;
    long int rayid;
    double val;

    fprintf(stdout, "reading vector b from '%s' ... ", filename);
    if (!(fd = fopen(filename, "r"))) {
        perror(filename);
        exit(1);
    }

    nb_read = fscanf(fd, "%ld", &n);
    if (nb_read != 1) {
        fprintf(stderr, "Error reading 'n' in '%s'\n", filename);
        exit(1);
    }
    fprintf(stdout, "size is %ld ... ", n);
    fflush(stdout);

    b = new_vector(n);

    /* load vector data from file */
    j = 0;
    while (!feof(fd)) {
        nb_read = fscanf(fd, "%ld %lf\n", &rayid, &val);

        if (nb_read != 2) {
            fprintf(stderr, "Error reading mat[%ld] in '%s'\n",
                    j, filename);
            exit(1);
        }

        if (fabs(b->mat[rayid]) > 1.0e-6) {
            fprintf(stdout,
                    "read_vector: duplicate value (%ld) old=%f/new=%f\n",
                    rayid, b->mat[rayid], val);
        }
        b->mat[rayid] = val;
        j++;
    }
    fclose(fd);
    fprintf(stdout, "%ld lines\n", j);
    fflush(stdout);
    return (b);
}



void write_vector(struct vector_t *b, char *filename)
{
    long int i;
    FILE *fd;

    fprintf(stdout, "writing vector to '%s' ... ", filename);
    if (!(fd = fopen(filename, "w"))) {
        perror(filename);
        exit(1);
    }

    fprintf(fd, "%ld\n", b->length);
    for (i = 0; i < b->length; i++) {
/*        fprintf(fd, "%ld %f\n", i, b->mat[i]);      */
        fprintf(fd, "%ld %14.10f\n", i, b->mat[i]);

    }

    fclose(fd);
    fprintf(stdout, "%ld items\n", b->length);
}

/** \brief read a portion of a vector 
 *
 * very specific, used by ray2mesh to re-number the residuals.
 */
struct vector_t *read_subvector(char *filename, long int *first,
                                long int *last)
{
    struct vector_t *b;
    long int n, j;
    FILE *fd;
    int nb_read;
    long int rayid, min_rayid, max_rayid;
    double val;

    fprintf(stdout, "reading sub-vector b from '%s' ... ", filename);
    if (!(fd = fopen(filename, "r"))) {
        perror(filename);
        exit(1);
    }

    nb_read = fscanf(fd, "%ld", &n);
    if (nb_read != 1) {
        fprintf(stderr, "Error reading 'n' in '%s'\n", filename);
        exit(1);
    }

    b = new_vector(n);
    min_rayid = n;
    max_rayid = 0;

    /* load vector data from file */
    j = 0;
    while (1) {
        nb_read = fscanf(fd, "%ld %lf", &rayid, &val);

        if (feof(fd)) {
            break;
        }

        if (nb_read != 2) {
            fprintf(stderr, "Error reading mat[%ld] in '%s'\n",
                    j, filename);
            exit(1);
        }

        if (max_rayid < rayid) {
            max_rayid = rayid;
        }
        if (min_rayid > rayid) {
            min_rayid = rayid;
        }

        b->mat[rayid] = val;
        j++;
    }
    fclose(fd);

    *first = min_rayid;
    *last = max_rayid;

    b->length = max_rayid + 1;
    b->mat = (double *) realloc(b->mat, sizeof(double) * b->length);

    fprintf(stdout, "%ld lines\n", j);
    fflush(stdout);
    return (b);
}

struct vector_t *vector_resize(struct vector_t *v, long int new_length)
{
    long int old_length, i;

    if (!v) {
        return (NULL);
    }
    old_length = v->length;
    v->mat = (double *) realloc(v->mat, sizeof(double) * new_length);
    assert(v->mat);
    v->length = new_length;
    for (i = old_length; i < new_length; i++) {
        v->mat[i] = 0;
    }

    fprintf(stdout, "vector_resize: resize (%p) from %ld to %ld\n",
            v, old_length, new_length);
    return (v);
}




// ___________________________________________
// CZ 10 fevrier 2020

struct vector_t *read_simple_vector_withYkreconstructedfromTk(char * filename_nomVecteur_Tk,double * tab_FULLFirstRowOfGmatrix, double * tab_allVj, double max_Vj, int normVj, double * tab_all_ci, double damping, long int nb_para) 
{

// ici les Vj ne sont pas normalises par Vjmax

    struct vector_t *b;
    long int  j;
    FILE *fd;
    double Tkj_val;

    fprintf(stdout, "reading 'simple' vector b from '%s' ... ", filename_nomVecteur_Tk);
    if (!(fd = fopen(filename_nomVecteur_Tk, "r"))) {
        perror(filename_nomVecteur_Tk);
        exit(1);
    }

printf("\nnb_para=%ld\n",nb_para);

  //  nb_para represente les M valeurs du fichier Tk (M est le nombre de paramètres du modèle)



    b = new_vector(nb_para+1); // ATTENTION : ici je fais +1 car le vecteur Yk va avoir M+1 elements

int nb_read;

    /* load vector data from file */
    j = 0;
    while (!feof(fd))
   {
        nb_read = fscanf(fd, "%lf\n", & Tkj_val);
	if (nb_read != 1)
	   {
	    fprintf(stderr, "Error reading mat[%ld] in '%s'\n",j, filename_nomVecteur_Tk);
            exit(1);
	    }

        // b->mat[j] = Tkj_val * sqrt(tab_allVj[j]) - (double) (1./tab_all_ci[0]) * (double) (tab_FULLFirstRowOfGmatrix[j] / sqrt(tab_allVj[j]) ) ;    // CZ 03/02/2020--------------------
        if (normVj == 0)
        b->mat[j] = Tkj_val * sqrt(tab_allVj[j]*max_Vj) - (double) (1./tab_all_ci[0]) * (double) (tab_FULLFirstRowOfGmatrix[j] / sqrt(tab_allVj[j] / max_Vj) ) ;    // CZ 03/02/2020--------------------
        else
        b->mat[j] = Tkj_val * sqrt(tab_allVj[j]*max_Vj) - (double) (1./tab_all_ci[0]) * (double) (tab_FULLFirstRowOfGmatrix[j]) ;    // CZ 03/02/2020--------------------
           
        j++;
    }


    
       if (j != nb_para)
       {
        fprintf(stderr,
                "read_simple_vector: truncated file ! read only %ld/%ld items.\n",
                j, nb_para);
        exit(1);
       }
  
        
       b->mat[j] = - (double) (1./tab_all_ci[0]) * damping ;  // dernier element du vecteur y_k,  qu'on doit rajoute a la fin (d'ou le n+1 d'avant) -- CZ 03/02/2020--------------------
    

       printf("\n le vecteur yk contient %ld lignes", j);

    fclose(fd);
    fprintf(stdout, "%ld lines\n", j);
    fflush(stdout);
    return (b);
}
//________________________________________



// 03/02/2020
// C.Z.



void write_vector__with_xk_reconstructed_from_xtildek(struct vector_t *b, char *filename, double * tab_all_ci)
{
    long int i;
    FILE *fd;

    fprintf(stdout, "writing vector to '%s' ... ", filename);
    if (!(fd = fopen(filename, "w"))) {
        perror(filename);
        exit(1);
    }

    double val= (double) 1./tab_all_ci[0];


    for (i = 0; i < b->length; i++)
        val -= (double)(tab_all_ci[i+1]/tab_all_ci[0]) * (b->mat[i]) ; 


    
//    fprintf(fd, "%ld\n", b->length);


   // on rajoute la premiere valeur au debut


    fprintf(fd, "%14.10f\n",val);


  for (i = 0; i < b->length; i++)
        fprintf(fd, "%14.10f\n", b->mat[i]);



    fclose(fd);
}

// CZ lundi 10 fevrier 2020

struct vector_t *read_simple_vector__WITHOUTHEADER_CZ(char *filename, long int nb_data)
{   
    struct vector_t *b;
    long int  i;
    FILE *fd;
    int nb_read;
    double val;
    
    fprintf(stdout, "reading 'simple' vector b from '%s' ... ", filename);
    if (!(fd = fopen(filename, "r"))) {
        perror(filename);
        exit(1);
    }

    
    b = new_vector(nb_data);
    
    /* load vector data from file */
    i = 0;
    while (!feof(fd)) {
        nb_read = fscanf(fd, "%lf\n", &val);
        if (nb_read != 1) { 
            fprintf(stderr, "Error reading mat[%ld] in '%s'\n",
                    i, filename);
            exit(1);
        }
        b->mat[i] = val;
        i++;
    }
    
    if (i != nb_data) {
        fprintf(stderr,
                "read_simple_vector: truncated file ! read only %ld/%ld items.\n",
                i, nb_data);
        exit(1);
    }
    
    fclose(fd);
    fprintf(stdout, "%ld lines\n", i);
    fflush(stdout);
    return (b);
}

// 10/02/2020
// C.Z.



void write_vector__with_Ak_reconstructed_from_Vj(struct vector_t *b, char *filename, double * tab_allVj)
{
    long int j;
    FILE *fd;

    fprintf(stdout, "writing vector to '%s' ... ", filename);
    if (!(fd = fopen(filename, "w"))) {
        perror(filename);
        exit(1);
    }

    for (j = 0; j < b->length; j++)
        fprintf(fd, "%14.10f\n", (double)(b->mat[j])/(tab_allVj[j]));

    fclose(fd);
}


void write_vector_Ak_reconstructed_from_Vj(struct vector_t *b, char *filename, double * factnorm)
{
    long int j;
    FILE *fd;

    fprintf(stdout, "writing vector to %s ... ", filename);
    fprintf(stdout, "%s\n", filename);
    if (!(fd = fopen(filename, "w"))) {
        perror(filename);
        exit(1);
    }

    for (j = 0; j < b->length; j++)
        fprintf(fd, "%14.10f\n", (double)(b->mat[j])*factnorm[j]);

    fclose(fd);
}


void write_matrix_R_reconstructed_from_Vj(FILE *fd, long int k, struct vector_t *b, double * factnorm)
{
    long int j;

    for (j = 0; j < b->length; j++)
        if (b->mat[j] != 0.) 
            fprintf(fd, "%ld %ld %14.10f\n", k, j, (double)(b->mat[j])*factnorm[j]);

}
