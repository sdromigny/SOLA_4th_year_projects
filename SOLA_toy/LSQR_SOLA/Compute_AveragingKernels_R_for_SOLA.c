// Compute_AveragingKernels_R_for_SOLA_BGladTomo.c -- Christophe Zaroli -- University of Strasbourg, 2020

/*
pour tester, lancer la commande suivante :

Compute_AveragingKernels_R_for_SOLA_BGladTomo TRANSPOSED_G_MATRIX_NORMALIZED_BY_Vj_AND_SIGMAdi.txt GENERALISED_INVERSE_filename_xk indice_k_min indice_k_max AVERAGING_KERNEL_Ak TestingBGlad__allVj.txt Rcomp normVj

TRANSPOSED_G_MATRIX_NORMALIZED_BY_Vj_AND_SIGMAdi.txt
--> contient :
 nbr de lignes du fichier
 nbr de lignes de G^T (c'est a dire M, le nbr de paramètres du modele) et nbr de colonnes de G^T (c'est a dire N, le nbr de données)
 indice_ligne indice_colonne valeur_non_nulle
 .             .                .
 .             .                .
 .             .                .
 .             .                .
 .             .                .
 
 (comme d'habitude, il faut que ce fichier soit trié d'abord sur les lignes, puis sur les colonnes)
 
 GENERALISED_INVERSE_filename_xk 
 --> les fichiers GENERALISED_INVERSE_filename_xk_{indice_k}.txt avec indice_k \in [indice_k_min, indice_k_max]
     sont les fichiers qui sortent du code LSQR_inversion_for_SOLA_BGladTomo.c
     et donc, a present, ils n'ont plus d'entete !!!
 
 AVERAGING_KERNEL_Ak
 --> ce code va ecrire des fichiers texte du type AVERAGING_KERNEL_Ak_{indice_k}.txt avec indice_k \in [indice_k_min, indice_k_max]
     ou un fichier du type AVERAGING_KERNEL_Ak.txt contenant la matrice de resolution si Rcomp = 1 
 
 TestingBGlad__allVj.txt
 --> ce fichier est a present utilise pour sortir directement les valeurs A^{k}_{j}
     c'est a dire A^{k}_{j} <--  PRODUCT(TRANSPOSED_G_MATRIX_NORMALIZED_BY_Vj_AND_SIGMAdi.txt ,  GENERALISED_INVERSE_filename_xk_{indice_k}.txt)
                  A^{k}_{j} <-- DIVIDE(A^{k}_{j}, TestingBGlad__allVj)

 Rcomp
 --> 0 : calcul des Ak (fichier individuel pour chaque Ak)
     1 : calcul de la matrice C (1 seule fichier : ligne 1 : nb lignes, nb colonnes ; autres lignes : i, j, val)

 normVj
 --> 0 : G transpose is not normalized by sqrt(Vj/max_Vj)
     1 : G transpose is normalised by sqrt(Vj/max_Vj)
 
*/


#define BUFFMAX 80 		// taille buffer lecture fichier donnees


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <signal.h>

//#include <sparse/sparse.h>
#include "fzsparse.h"
#include <time.h>

#include "lsqr.h"
#include "lsqr_wrapper.h"
#include "catch_sig.h"
#include "extern.h"

/* prod : compute A.X product
          this is just a test/example using sparse matrix (m x n) operator

		usage  : FZ_prod_multi matrixfile nomGen_VectorFile indiceDeb indiceFin nonGen_SolutionFile


		resolution d'un ensemble de fichiers de donnees defini par une plage de valeurs
		//
		// le nonm des fichiers de donnees est  defini par
		//
		//      nomGen_VectorFile_num.txt  ou num appartient a la plage de valeurs [indiceDeb, indiceFin]
		//
		//  les fichiers resultats produits ont pour nom : <nonGen_SolutionFile_num.txt>



   utilisation de la structure de matrice creuse implémentée en mémoire 
avec une liste des elements par ligne et une liste des elements par colonnes

nb_item
val
...

*/

int please_stop_lsqr = 0;       /* stop lsqr and write the solution at the curent iteration */
int please_dump_lsqr = 0;       /* dump intermediate solution each  iterdump iteration */

/**********************************/
/* MAIN                           */
/**********************************/

int main(int argc, char *argv[])
{
   struct sparse_matrix_t *sparseA;
   struct vector_t *b;
   struct vector_t *rhs;       /* right hand side */
   int noDeb, noFin, noCour, Rcomp, normVj;
   double deb_t, fin_t; // pour mesure du temps d'execution

   deb_t = clock();   // debut mesure temps CPU
   
   /* catch Ctrl-C signal */
   signal(SIGINT, emergency_halt);

   /* cmd line arg */
   char *matrix_filename = NULL;
   char *vector_filename = NULL;
   char *sol_filename = NULL;
   
   char * allVj_filename = NULL;

   if (argc != 9) {
      fprintf(stderr, "%d\n", argc);
      fprintf(stderr, "%s matrixfile nomGen_VectorFile indiceDeb indiceFin nonGen_SolutionFile allVj Rcomp normVj\n",
                argv[0]);
      exit(1);
   }
   matrix_filename = strdup(argv[1]);
   vector_filename = strdup(argv[2]);
   noDeb = (int) strtol(argv[3], (char **) NULL, 10);
   noFin = (int) strtol(argv[4], (char **) NULL, 10);
   sol_filename = strdup(argv[5]);
   allVj_filename = strdup(argv[6]);
   Rcomp = (int) strtol(argv[7], (char **) NULL, 10);
   normVj = (int) strtol(argv[8], (char **) NULL, 10);


   /* read the sparse matrix */
   sparseA = read_ijk_fzsparse_matrix_Tas(matrix_filename);
   fprintf(stderr, "read*matrix: ok (size=%ldx%ld, %ld elements)\n",
            sparseA->nb_line, sparseA->nb_col,
            sparseA->nb_line * sparseA->nb_col);
   show_sparse_stats(sparseA);
   
   fprintf(stderr, "\ntemps Cpu pour chargement matrice : %f\n", (double)(clock()-deb_t) / (double)
                         CLOCKS_PER_SEC);

   rhs = new_vector(sparseA->nb_line);   // vecteur resultat 

   int TotalNumberOfData = (int) sparseA->nb_col;
   printf("TotalNumberOfData=%d",TotalNumberOfData);   

   
   int TotalNumberOfModelParameters = (int) sparseA->nb_line ;
   printf("TotalNumberOfModelParameters=%d",TotalNumberOfModelParameters);   

   // si la matrice R est calcule, noDeb = 0, noFin = M-1
   if (Rcomp == 1) {
      noDeb = 0;
      noFin = TotalNumberOfModelParameters-1;
      }

   //=========================================================
   // CZ 03-10/FEBRUARY/2020
   
   // lecture du fichier des Vj

   FILE *fichier=NULL;
   char buff[BUFFMAX + 1];
   int index_j;
   double valeur;
   double   *tab_allVj ;
   if((tab_allVj = ( double *) malloc ( (TotalNumberOfModelParameters  ) * sizeof(double))) < 0 )
         printf("\n erreur allocation memoire tab_allVj \n");

   fichier=fopen(allVj_filename,"r+");
   printf("%s",allVj_filename);

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

   // compute max_Vj = max(Vj) 
   // Vj not normalised by Vjmax

   double max_Vj =  tab_allVj[0];

   for (index_j=1; index_j<TotalNumberOfModelParameters; index_j++)
   {
       if (tab_allVj[index_j]> max_Vj)
	   max_Vj = tab_allVj[index_j];
   }

   // check reconstructed values
   for (index_j=0; index_j<10; index_j++)
	printf("\ntab_allVj[index_j]=%lf", tab_allVj[index_j]);

   // Compute normalisation factor for Ak or R (depending or normVj) 
   // if normVj == 0 : Gij is not normalised by sqrt(Vj/max_Vj)
   
   double   *factnorm;
   if((factnorm = ( double *) malloc ( (TotalNumberOfModelParameters  ) * sizeof(double))) < 0 )
         printf("\n erreur allocation memoire factnorm \n");

   for (index_j=0; index_j<TotalNumberOfModelParameters; index_j++)
       if (Rcomp == 0)
          if (normVj == 0)
              factnorm[index_j] = 1/tab_allVj[index_j] ;
          else
              factnorm[index_j] = 1/sqrt(tab_allVj[index_j]*max_Vj) ;
       else
          if (normVj == 0)
              factnorm[index_j] = 1. ;
          else
              factnorm[index_j] = sqrt(tab_allVj[index_j]/max_Vj) ;

   
//==========================Debut traitement ==============================

   FILE *fd=NULL;
   char * nomSolution = malloc(strlen(sol_filename+10));  
   if (Rcomp == 1) {
      sprintf(nomSolution, "%s",sol_filename);
      fprintf(stdout, "writing R matrix to '%s' ... ", nomSolution);
      if (!(fd = fopen(nomSolution, "w"))) {
          perror(nomSolution);
          exit(1);
      }
      else
         fprintf(fd, "%d %d\n",TotalNumberOfModelParameters, TotalNumberOfModelParameters);
   }

   for (noCour=noDeb; noCour <= noFin; noCour++)
   {
   
      // construction du nom du fichier = radical_noCour
      char * nomVecteur = malloc(strlen(vector_filename) + 10);  // pour avoir de la marge
      
      sprintf(nomVecteur, "%s_%d.txt",vector_filename,noCour);
      
    
      FILE *fidy=fopen(nomVecteur, "r");
      if( fidy == NULL )
      {
      // do nothing
      }
      else
      {

      printf("\nlecture fichier vecteur de donnees %s \n",nomVecteur);
   
      //b = read_simple_vector(nomVecteur);
      b = read_simple_vector__WITHOUTHEADER_CZ(nomVecteur,TotalNumberOfData); // CZ 10/02/2020
      
   
      /*************************************************/
      /* check compatibility between matrix and vector */
      /*  for product     Axb                          */
      /*************************************************/

      if (sparseA->nb_col != b->length) {
         fprintf(stderr,
                "Error, check your matrix/vector sizes (%ld/%ld) vecteur : %s\n",
                sparseA->nb_col, b->length,nomVecteur);
         exit(1);
      }
   
      fprintf(stderr, "Starting product ...\n");
   
      /* product */
      /*  re-use some code */
      /* compute if mode=0   rhs = rhs + A.b   */
      /* rhs is null */
   
      deb_t = clock();
   
      //rhs = new_vector(sparseA->nb_line);
      sparseMATRIXxVECTOR(0, (dvec *) b, (dvec *) rhs, sparseA);
   
      fprintf(stderr, "\ntemps Cpu pour le traitement: %f\n", (double)(clock()-deb_t) / (double)
                          CLOCKS_PER_SEC);
   
      //write_vector((struct vector_t *) rhs, nomSolution);
      if (Rcomp == 0) {
          fprintf(stdout, "Rcomp = %d\n", Rcomp);
          // construction du nom du vecteur solution
   
          char * nomSolution = malloc(strlen(sol_filename) + 10);
          sprintf(nomSolution, "%s_%d.txt",sol_filename,noCour);
          fprintf(stdout, "%s\n", nomSolution);
          write_vector_Ak_reconstructed_from_Vj((struct vector_t *) rhs, nomSolution, factnorm);
          }
      else
          write_matrix_R_reconstructed_from_Vj(fd, noCour, (struct vector_t *) rhs, factnorm);
     
      // on remet le vecteur rhs a 0
      raz_vector(rhs);
   
      fclose(fidy);
      }
   }  // fin de la  boucle de traitement de tous  les vecteurs donnees

// on libere la memoire
   free_fzsparse_matrix(sparseA);
 
   if (Rcomp == 1) 
       fclose(fd);

   return (1);
}
