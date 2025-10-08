#include "lsqr_wrapper.h"
#include <time.h>

extern long int _nb_appel;

// tas et index pour reorganisation des colonnes
extern long int _index_tas_col;
extern struct sparse_item_t * _tas_col;
extern const long int NB_ITEM_MATRIX ;
extern struct sparse_item_t * _matrix_tas;

const char *lsqr_msg[] = {
    "The exact solution is x = x0",
    "The residual Ax - b is small enough, given ATOL and BTOL",
    "The least squares error is small enough, given ATOL",
    "The estimated condition number has exceeded CONLIM",
    "The residual Ax - b is small enough, given machine precision",
    "The least squares error is small enough, given machine precision",
    "The estimated condition number has exceeded machine precision",
    "The iteration limit has been reached",
    "Interuption requested by the user (Ctrl-C)"
};

/*************************************/

/* computes :                        */

/* if mode = 0   ->  y = y + A.x   */

/* if mode = 1   ->  x = x + A^T.y */

/*************************************/
void sparseMATRIXxVECTOR(long int mode, dvec * x, dvec * y, void *data)
{
   struct sparse_matrix_t *a;
   long int i, k;

   a = (struct sparse_matrix_t *) data;

   if (mode == 0) {
        /* Compute  Y = Y + A*X */

       for(i=0; i < a->nb_item; i++){   
	    y->elements[a->_tas_l[i].line_index] += a->_tas_l[i].val * x->elements[a->_tas_l[i].col_index];
       }
      return;
   }

   if (mode == 1) {
        /* Compute  X = X + A^T*Y */
      
       for(i=0; i < a->nb_item; i++){   
	    x->elements[a->_tas_c[i].col_index] += a->_tas_c[i].val * y->elements[a->_tas_c[i].line_index];
       }
      
    }
}

//________________________________________________________________________________________
