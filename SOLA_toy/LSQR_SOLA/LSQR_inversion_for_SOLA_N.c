// LSQR_inversion_for_SOLA_BGladTomo.c -- Christophe Zaroli -- University of Strasbourg, 2020

/*
pour tester, lancer la commande suivante :

LSQR_inversion_for_SOLA_BGladTomo /home/christophe/ForBGladTomo__LSQRforSOLA/EXAMPLE_INPUTS/TestingBGlad_matrixfile_Q.txt /home/christophe/ForBGladTomo__LSQRforSOLA/EXAMPLE_INPUTS/TestingBGlad__vect
orfile_T 1000 1000 /home/christophe/ForBGladTomo__LSQRforSOLA/EXAMPLE_OUTPUTS/solutionfile_xk 100 50 /home/christophe/ForBGladTomo__LSQRforSOLA/EXAMPLE_INPUTS/TestingBGlad__all_ci.txt /hom
e/christophe/ForBGladTomo__LSQRforSOLA/EXAMPLE_INPUTS/TestingBGlad__Gfirstrow.txt /home/christophe/ForBGladTomo__LSQRforSOLA/EXAMPLE_INPUTS/TestingBGlad__allVj.txt 1 101 
*/


#define BUFFMAX 80 		// taille buffer lecture fichier donnees

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include <time.h>
#include "fzsparse.h"
#include "lsqr.h"
#include "lsqr_wrapper.h"
#include "catch_sig.h"
#include "extern.h"

/* NOUVEAUX ARGUMENTS (CZ 01/02/2020) :
0.LSQR_inversion_for_SOLA_BGladTomo
1.matrixfile_Q
2.vectorfile_T
3.numInf_kindex
4.numSup_kindex
5.solutionfile_xk
6.nbitermax_lsqr
7.damping_eta
8.all_ci
9.Gfirstrow
10.allVj
12.normVj
11.[iterdump_lsqr]
*/


/*
 *
 CZ 29/01/2020 :
 * je veux lire { les fichiers T^k_j, c_1, Gmatrix_firstrow, Vj } afin de reconstruire directement {les fichiers y^k_j}
 * et je veux aussi retourner directement l'inverse generalisee x^k !!!
 * cela rendra les choses plus simples et evitera plusieurs lecture/ecriture dans le code python !!!!!

 normVj
 --> 0 : Gfirstrow is not normalized by sqrt(Vj/max_Vj)
     1 : Gfirstrow is normalised by sqrt(Vj/max_Vj)
 */


/* sola_multi.c  :
resolution de systeme d'equations
utilisant la structure de matrice creuse implémentée en mémoire
avec une liste des elements par ligne et une liste des elements par colonnes
nb_item
m n
i j val
...
*/



int please_stop_lsqr = 0;       /* stop lsqr and write the solution at the curent iteration */
int please_dump_lsqr = 0;       /* dump intermediate solution each  iterdump iteration */






/********/
/* MAIN */
/********/
int main(int argc, char *argv[])
{
   struct sparse_matrix_t *sparseA; // *m,*w,*z,*t;
   struct vector_t *x, *b;

   lsqr_input *input;
   lsqr_output *output;
   lsqr_work *work;            /* zone temoraire de travail */
   lsqr_func *func;            /* func->mat_vec_prod -> APROD */

   int noDeb, noFin, noCour;

   // pour mesure du temps d'execution

   double deb_t, fin_t; // pour mesure du temps d'execution

   deb_t = clock();   // debut mesure temps CPU

   // lecture de la matrice creuse par le processus maitre

   /* cmd line arg */
   char * matrix_filename_Q = NULL;
   char * vector_filename_T = NULL;
   char *sol_filename = NULL;
   int max_iter = -1;
   int normVj = 0;
   
   char * all_ci_filename = NULL;
   char * Gfirstrow_filename = NULL;
   char * allVj_filename = NULL;

   double damping ; 
   //________________________________________________
  


   if (argc != (8+4) && argc != (9+4)) // +3 for c_1, Gfirstrow, allVj, normVj (et je remets  iterdump en dernier)
   {
      fprintf(stderr,
             "%s 1.matrix_filename_Q 2.vector_filename_T 3.numInf_kindex 4.numSup_kindex 5.solutionfile_xk 6.nbitermax_lsqr 7.damping_eta 8.all_ci 9.Gfirstrow 10.allVj 11.normVj 12.[iterdump_lsqr] \n",
             argv[0]);
      exit(1);
   }
   matrix_filename_Q = strdup(argv[1]);
   vector_filename_T = strdup(argv[2]);
   noDeb = (int) strtol(argv[3], (char **) NULL, 10);
   noFin = (int) strtol(argv[4], (char **) NULL, 10);
   sol_filename = strdup(argv[5]);
   max_iter = (int) strtol(argv[6], (char **) NULL, 10);
   damping = strtod(argv[7], (char **) NULL); // pour SOLA --> damping==eta
   
   //_______________________
   all_ci_filename = strdup(argv[8]); 
   Gfirstrow_filename = strdup(argv[9]);
   allVj_filename = strdup(argv[10]);
   normVj = (int) strtol(argv[11], (char **) NULL, 10);
   //_______________________
  
           
   if (argc == (9+4)) {
      please_dump_lsqr = (int) strtol(argv[12], (char **) NULL, 10);
   }
   else {
      please_dump_lsqr = 0;
   }
   /* read the sparse matrix */
   //sparseA = read_ijk_sparse_matrix(matrix_filename_Q, SPARSE_COL_LINK);
   sparseA = read_ijk_fzsparse_matrix_Tas(matrix_filename_Q);
   fprintf(stderr, "read*matrix: ok (size=%ldx%ld, %ld elements)\n",
          sparseA->nb_line, sparseA->nb_col,
          sparseA->nb_line * sparseA->nb_col);
   show_sparse_stats(sparseA);


   /* catch Ctrl-C signal */
   signal(SIGINT, emergency_halt);


//=========================================================
// a faire au debut du traitement pour tous les vecteurs
   /* LSQR alloc */
   alloc_lsqr_mem(&input, &output, &work, &func,
                 sparseA->nb_line, sparseA->nb_col);

   fprintf(stderr, "alloc_lsqr_mem : ok\n");

   /* defines the routine Mat.Vect to use */
   func->mat_vec_prod = sparseMATRIXxVECTOR;

   /* Set the input parameters for LSQR */
   input->num_rows = sparseA->nb_line;
   input->num_cols = sparseA->nb_col;
   input->rel_mat_err = .0;
   input->rel_rhs_err = .0;
   input->cond_lim = .0;
   input->lsqr_fp_out = stdout;
      //input->rhs_vec = (dvec *) b;      // specifique a chaque vecteur de donnees
      //input->sol_vec = (dvec *) x;        /* initial guess */
   input->damp_val = damping;
   if (max_iter == -1) {
      input->max_iter = 4 * (sparseA->nb_col);
   }
   else {
      input->max_iter = max_iter;
   }
//=========================================================


      fin_t = clock();
      fprintf(stderr, "\ntemps Cpu chargement matrice : %f ", (double)(fin_t-deb_t) / (double)
                 CLOCKS_PER_SEC);


//=========================================================
// CZ 03/02/2020
// lecture du fichier Gfirstrow_filename et reconstruction de tab_FULLFirstRowOfGmatrix

   FILE *fichier=NULL;
   char buff[BUFFMAX + 1];
   int index_j;
   double valeur;

   int TotalNumberOfModelParameters = (int) sparseA->nb_line ;
   TotalNumberOfModelParameters = TotalNumberOfModelParameters -1;
  int TotalNumberOfData = (int) sparseA->nb_col;
   TotalNumberOfData=TotalNumberOfData+1;

   // car le nombre de lignes de la matrice Q est M+1, et je voulais récupérer la valeur de M (nbr de paramètres)
   // et car le nombre de colonnes de la matrice Q est N-1, et je voulais recuperer la valeur de N (nbr de données)
   //
   printf("\nTotalNumberOfModelParameters=%d, TotalNumberOfData=%d\n", TotalNumberOfModelParameters,TotalNumberOfData);
  
   double   *tab_FULLFirstRowOfGmatrix ;
   if((tab_FULLFirstRowOfGmatrix = ( double *) malloc ( (TotalNumberOfModelParameters  ) * sizeof(double))) < 0 )
         printf("\n erreur allocation memoire tab_FULLFirstRowOfGmatrix \n");

// initialisation
 for (index_j=0; index_j<TotalNumberOfModelParameters; index_j++)
	tab_FULLFirstRowOfGmatrix[index_j]=0.;

//printf("\n%s\n",Gfirstrow_filename);

   fichier=fopen(Gfirstrow_filename,"r+");

   if (fichier != NULL)
   {
    // printf("\nOn peut lire et ecrire dans le fichier\n");
   
      while(fgets(buff,BUFFMAX,fichier) != NULL)
      {
         sscanf(buff, "%d %lf", &index_j, &valeur); tab_FULLFirstRowOfGmatrix[index_j]=valeur;
      
      }
      fclose(fichier);

   }
   else
   {
      printf("\nImpossible d'ouvrir le fichier\n");
   }

// check reconstructed values
 for (index_j=0; index_j<10; index_j++)
	printf("\ntab_FULLFirstRowOfGmatrix[index_j]=%lf",tab_FULLFirstRowOfGmatrix[index_j]);


//=========================================================

//=========================================================
// CZ 03/02/2020
// lecture du fichier des Vj


   double   *tab_allVj ;
   if((tab_allVj = ( double *) malloc ( (TotalNumberOfModelParameters  ) * sizeof(double))) < 0 )
         printf("\n erreur allocation memoire tab_allVj \n");

   fichier=fopen(allVj_filename,"r+");

   if (fichier != NULL)
   {
    // printf("\nOn peut lire et ecrire dans le fichier\n");
      index_j=0;
      while(fgets(buff,BUFFMAX,fichier) != NULL)
      {
         sscanf(buff, "%lf",&valeur);
	 tab_allVj[index_j]=valeur;
         index_j++; 
      }
      fclose(fichier);

   }
   else
   {
      printf("\nImpossible d'ouvrir le fichier\n");
   }

// je divise les Vj par max(Vj) -- Remarque : tous les Vj sont > 0 par definition

double max_Vj =  tab_allVj[0];

for (index_j=1; index_j<TotalNumberOfModelParameters; index_j++)
{
	if (tab_allVj[index_j]> max_Vj)
		max_Vj = tab_allVj[index_j];
}

// je normalise tous les Vj par max_Vj;

//for (index_j=0; index_j<TotalNumberOfModelParameters; index_j++)
//	tab_allVj[index_j] /= max_Vj ;



// check reconstructed values
// for (index_j=0; index_j<10; index_j++)
//	printf("\ntab_allVj[index_j]=%lf", tab_allVj[index_j]);

//=========================================================

//=========================================================
// CZ 03/02/2020
// lecture du fichier des ci (the index i ranges from 0 to N-1, with N the total nor of data)


   double   *tab_all_ci ;
   int index_i=0;
   if((tab_all_ci = ( double *) malloc ( (TotalNumberOfData  ) * sizeof(double))) < 0 )
         printf("\n erreur allocation memoire tab_all_ci \n");


//printf("\n%s\n",all_ci_filename);

   fichier=fopen(all_ci_filename,"r+");

   if (fichier != NULL)
   {
    // printf("\nOn peut lire et ecrire dans le fichier\n");
   
      while(fgets(buff,BUFFMAX,fichier) != NULL)
      {
         sscanf(buff,"%lf", &valeur);
	 tab_all_ci[index_i]=valeur;
	 index_i++;
      
      }
      fclose(fichier);

   }
   else
   {
      printf("\nImpossible d'ouvrir le fichier\n");
   }

// check reconstructed values
 for (index_i=0; index_i<10; index_i++)
	printf("\n tab_all_ci[index_i]=%lf", tab_all_ci[index_i]);

//=========================================================




   for (noCour=noDeb; noCour <= noFin; noCour++)
   {


   deb_t = clock();   // debut mesure temps CPU



   // construction du nom du fichier = radical_noCour
      char * nomVecteur_Tk = malloc(strlen(vector_filename_T) + 10);  // pour avoir de la marge

      sprintf(nomVecteur_Tk, "%s_%d.txt", vector_filename_T,noCour);


printf("\n%s\n",nomVecteur_Tk);


// TO DO

// juste tester ICI si le fichier donnee  existe, sinon on passe au suivant
// ici je pourrais lire un autre fichier qui me dirait si je veux effectivement
// calculer la solution sola pour le point (k) en question  [CZ 29/01/20]


FILE *fidy=fopen(nomVecteur_Tk, "r");
if( fidy == NULL )
{
// do nothing
}
else
{

    // construction du nom du vecteur solution (x_k donc a present)

      char * nomSolution = malloc(strlen(sol_filename) + 10);
      sprintf(nomSolution, "%s_%d.txt",sol_filename,noCour);


      printf("\nlecture fichier de donnees %s \n", nomVecteur_Tk);

      // b = read_simple_vector(nomVecteur);  
  
      //_______________________ CZ 03/02/2020

       b = read_simple_vector_withYkreconstructedfromTk(nomVecteur_Tk,tab_FULLFirstRowOfGmatrix,tab_allVj,max_Vj,normVj,tab_all_ci,damping,TotalNumberOfModelParameters); 
      
      

      printf("\nFin lecture vecteur  donnees ");
   /*************************************************/
   /* check compatibility between matrix and vector */
   /*************************************************/
      if (sparseA->nb_line != b->length) {
         fprintf(stderr,
             "Error, check your matrix/vector sizes (%ld/%ld)\n",
             sparseA->nb_line, b->length);
         exit(1);
      }
   /* init vector solution to zero */
      x = new_vector(sparseA->nb_col);

   /* mise a jour de la structure input avec le vecteur de donnees courant*/
      input->rhs_vec = (dvec *) b;      // specifique a chaque vecteur de donnees
      input->sol_vec = (dvec *) x;        /* initial guess */


   //    /* catch Ctrl-C signal */
   //       signal(SIGINT, emergency_halt);

   /*************************************************************/
   /* solve A.x = B                                             */
   /*************************************************************/

   //_______________________________________________
   //   /* LSQR alloc */
   //       alloc_lsqr_mem(&input, &output, &work, &func,
   //                  sparseA->nb_line, sparseA->nb_col);
   //
   //       fprintf(stderr, "alloc_lsqr_mem : ok\n");
   //
   //    /* defines the routine Mat.Vect to use */
   //       func->mat_vec_prod = sparseMATRIXxVECTOR;
   //
   //    /* Set the input parameters for LSQR */
   //       input->num_rows = sparseA->nb_line;
   //       input->num_cols = sparseA->nb_col;
   //       input->rel_mat_err = .0;
   //       input->rel_rhs_err = .0;
   //       input->cond_lim = .0;
   //       input->lsqr_fp_out = stdout;
   //       input->rhs_vec = (dvec *) b;
   //       input->sol_vec = (dvec *) x;        /* initial guess */
   //       input->damp_val = damping;
   //       if (max_iter == -1) {
   //          input->max_iter = 4 * (sparseA->nb_col);
   //       }
   //       else {
   //          input->max_iter = max_iter;
   //       }
    //_______________________________________________

   /* resolution du systeme Ax=b */
      lsqr(input, output, work, func, sparseA);

//      write_vector((struct vector_t *) output->sol_vec, nomSolution);

      write_vector__with_xk_reconstructed_from_xtildek((struct vector_t *) output->sol_vec, nomSolution, tab_all_ci);


   /******************************************************/
   /* variance reduction, ie how the model fits the data */
   /* X = the final solution                             */
   /*                                                    */
   /*                 ||b-AX||²                          */
   /*         VR= 1 - --------                           */
   /*                  ||b||²                            */
   /*                                                    */
   /******************************************************/
      {
         double norm_b;
         double norm_b_AX;
         double VR;              /* variance reduction */

         struct vector_t *rhs;   /* right hand side */
         rhs = new_vector(sparseA->nb_line);

      /* use copy */
         dvec_copy((dvec *) b, (dvec *) rhs);

         norm_b = dvec_norm2((dvec *) rhs);

      /* does rhs = rhs + sparseA . output->sol_vec */
      /* here  rhs is overwritten */
         dvec_scale((-1.0), (dvec *) rhs);
         sparseMATRIXxVECTOR(0, output->sol_vec, (dvec *) rhs, sparseA);
         dvec_scale((-1.0), (dvec *) rhs);

         norm_b_AX = dvec_norm2((dvec *) rhs);

      //VR = 1 - (norm_b_AX*norm_b_AX)/(norm_b*norm_b);
         VR = 1 - norm_b_AX / norm_b;
         fprintf(stdout, "Variance reduction = %.2f%%\n", VR * 100);
         free_vector(rhs);
      }

      //free_lsqr_mem(input, output, work, func);

   /* check A^t.A */
   /*
    * { struct sparse_matrix_t *AtA; AtA = AtransA (sparseA);
    * write_sparse_matrix(AtA, "AtA"); write_sparse_matrix(sparseA,
    * "A"); free_sparse_matrix (AtA);
    *
    * } */

      fin_t = clock();
      fprintf(stderr, "\ntemps Cpu du calcul d'un point (k) : %f ", (double)(fin_t-deb_t) / (double)
                 CLOCKS_PER_SEC);

   fclose(fidy);
}// fin du test si le fichier donnee "k" existe

}  // fin de la  boucle de traitement de tous  les vecteurs donnees
//======================================================================
   free_fzsparse_matrix(sparseA);
   return (1);
}




