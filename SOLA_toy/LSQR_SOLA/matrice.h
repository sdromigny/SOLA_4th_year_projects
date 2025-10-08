#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#ifndef __MATRICE_H__
#define __MATRICE_H__

struct matrix_t {
    long int nb_line;
    long int nb_col;
    double **mat;
};

struct vector_t {
    long int length;
    double *mat;
};

// CZ 10/02/2020
void raz_vector(struct vector_t *v);


struct matrix_t *new_matrix(int nb_line, int nb_col);
void free_matrix(struct matrix_t *c);

struct vector_t *new_vector(long int l);
void free_vector(struct vector_t *v);

void dump_matrix(char *txt, struct matrix_t *m);
void dump_vector(char *s, struct vector_t *v);

struct matrix_t *read_matrix(char *filename);
struct vector_t *read_vector(char *filename);
struct vector_t *read_simple_vector(char *filename);
// CZ 03/02/2020
struct vector_t *read_simple_vector_withYkreconstructedfromTk(char * filename_nomVecteur_Tk,double * tab_FULLFirstRowOfGmatrix, double * tab_allVj, double max_Vj, int normVj, double * tab_all_cj, double damping, long int nb_para); 
//struct vector_t *read_simple_vector_withYkreconstructedfromTk(char * filename_nomVecteur_Tk,double * tab_FULLFirstRowOfGmatrix, double * tab_allVj, double max_Vj, double * tab_all_cj, double damping, long int nb_para); 
void write_vector__with_xk_reconstructed_from_xtildek(struct vector_t *b, char *filename, double * tab_all_cj);


struct vector_t *read_subvector(char *filename, long int *first,
                                long int *last);
struct vector_t *import_vector(struct vector_t *b, char *filename);
struct vector_t *vector_resize(struct vector_t *v, long int new_length);
void write_vector(struct vector_t *b, char *filename);

// CZ 10/02/2020
struct vector_t *read_simple_vector__WITHOUTHEADER_CZ(char *filename, long int nb_para);
void write_vector__with_Ak_reconstructed_from_Vj(struct vector_t * rhs, char * nomSolution,  double * tab_allVj);

void write_vector_Ak_reconstructed_from_Vj(struct vector_t *b, char *filename, double * factnorm);
void write_matrix_R_reconstructed_from_Vj(FILE *fd, long int k, struct vector_t *b, double * factnorm);

#endif
