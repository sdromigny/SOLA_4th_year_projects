#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "matrice.h"

#ifndef __SPARSE_H__
#define __SPARSE_H__

#define EPS_SPARSE 1.0e-6
#define NOVALUE 0

//				Version sparse + tas + indice
enum {
   SPARSE_COL_LINK = 1
};

// structure pour les elements des deux tas memorisant le liste des items par ligne et pas colonne
// chaque maillon memorise le lien vers le maillon precedant dans sa ligne et sa colonne

struct sparse_item_t {
   long int col_index;
   long int line_index;
   double val;
   //struct sparse_item_t *prev_in_line;  // pointeur vers maillon precedant dans la ligne
   //struct sparse_item_t *prev_in_col;   // pointeur vers maillon precedant dans la colonne
   long int prev_in_line;  // indice dans _tas_l vers maillon precedant dans la ligne
   long int prev_in_col;   // indice dans _tas_c vers maillon precedant dans la colonne
};

//______________________________________________________________________________
// structure decrivant une matrice creuse
//
struct sparse_matrix_t {
   long int nb_line;
   long int nb_col;
   long int nb_item;
   //struct sparse_item_t **line;  // lien vers un tableau de pointeur donnant le dernier elt de la ligne
   //struct sparse_item_t **col;	// lien vers tableau de pointeur donnant le  dernier eet de la colonne

   long int *line;  // lien vers un tableau d'entiers donnant l'indice dans _tas_l du dernier elt de la ligne
   long int *col;   // lien vers tableau d'entiers donnant l'indice dans _tas_c du   dernier eet de la colonne


   struct sparse_item_t *_tas_l;	// pointeur vers le tableau  en MC contenant les listes d'item par ligne
   struct sparse_item_t *_tas_c;	// pointeur vers le tableau  en MC contenant les listes d'item par colonne 

};




char *libsparseversion();

struct sparse_matrix_t *new_fzsparse_matrix(long int nb_line, long int nb_col, long int nb_item);
//________________ FZ 07/10/2015

struct sparse_matrix_t *sparse_matrix_dupliquer_Tas(struct sparse_matrix_t *m); // variante avec tas                                 
struct sparse_item_t *sparse_set_value_Tas(struct sparse_matrix_t *m,
                                       long int i, long int j, double val,
                                       struct sparse_item_t *previous);
void affiche_item(struct sparse_item_t *t);
void affiche_sparse_m(struct sparse_matrix_t *m);

void free_sparse_matrix_Tas(struct sparse_matrix_t *m);
long int sparse_matrix_compter_ligne(struct sparse_matrix_t *m)  ;  // compte le nbre d'elements non nuls ligne/ligne
long int sparse_matrix_compter_colonne(struct sparse_matrix_t *m)  ;  // compte le nbre d'elements non nuls ligne/ligne

struct sparse_matrix_t *read_ijk_fzsparse_matrix_Tas(char *filename);  // lecture matrice avec structure fzsparse + Tas
double sparse_fz_get_value(struct sparse_matrix_t *m, long int i, long int j);
double sparse_fz_get_value(struct sparse_matrix_t *m, long int i, long int j);

void afficherLigne(struct sparse_matrix_t *m, long int nol, long int rangDeb,long int nb);

//__________________________________________________
void free_fzsparse_matrix(struct sparse_matrix_t *m);

/*
void dump_sparse_matrix_to_scilab(struct sparse_matrix_t *m);

struct vector_t *sparse_extract_col(struct sparse_matrix_t *A, long int c);
struct vector_t *sparse_extract_line(struct sparse_matrix_t *A,
                                    long int l);

struct sparse_matrix_t *sparsify(struct matrix_t *M, int col_link_status);

struct sparse_matrix_t *sparse_matrix_resize(struct sparse_matrix_t *m,
                                            long int nbline,
                                            long int nbcol);

int check_sparse_matrix(struct sparse_matrix_t *m);
void sparse_compute_length(struct sparse_matrix_t *m, char *filename);
struct sparse_matrix_t *AtransA(struct sparse_matrix_t *A);
double mean_diag_AtA(struct sparse_matrix_t *A);
*/
void show_sparse_stats(struct sparse_matrix_t *A);
#endif
