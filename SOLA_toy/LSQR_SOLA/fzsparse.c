#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "fzsparse.h"
//                   23/10/2015   FZ
//
// structure avec chainage inverse et allocation en memoire 
// de deux tas pour liste des items par ligne et liste des item par colonne
// accelere le traitement dans lsqr_wraper ainsi que le cahrgement
// de la matrice
//
// Remarque : le fichier contenant la liste des  vitems de al matrice doit commencer
// par le nombre total d'items ( afin d'allouer en MC les deux tableaux _tas_l et _tas_c

/** \brief Library information **/
char *libsparseversion()
{
   char *s;

   s = "PROVISOIRE"; // malloc(strlen(PACKAGE) + strlen(VERSION) + 3);
   sprintf(s, "%s-%s", "  ***", "***" ); //PACKAGE, VERSION);
   return (s);
}

//__________________________________________________
// compte le nombre de valeurs non nulles dans la matrice creuse
// avec chainage arriere ( parcours ligne/ligne)
long int sparse_matrix_compter_ligne(struct sparse_matrix_t *m)  
{
   long int i,cpt=0,nb=m->nb_line;
   long int l;
   
   for(i=0; i < m->nb_line ; i++)
   {
      l = m->line[i];
   
      if( l != -1)
      {
      // on compte les items de la ligne
      
         while( l != -1 ) 
         {
            cpt++;     
            l = m->_tas_l[l].prev_in_line; 
         }
      }
   } 	

   return cpt;
}   

//______________________________________________________________
// compte le nombre de valeurs non nulles dans la matrice creuse
// avec chainage arriere (parcours colonne/colonne)

long int sparse_matrix_compter_colonne(struct sparse_matrix_t *m)
{
   long int j,cpt=0;
   long int l;

   for(j=0; j < m->nb_col; j++)
   {
      l = m->col[j];
      if(l != -1 ) 
      {
         while(l != -1)
         {
            cpt++;
            l = m->_tas_c[l].prev_in_col; 
         }
      
      }
   }
   return cpt;
}
                              
//______________________________________________________________________
//  
//  affiche nb valeurs de la ligne nol en partant de la valeur de rangDeb depuis la fin de la ligne
//______________________________________________________________________

void afficherLigne(struct sparse_matrix_t *m, long int nol, long int rangDeb,long int nb)
{
   long int i,e;

   printf("\nAffichage valeurs de la ligne nol ");

   e = m->line[nol];

     // positionnement  au bon debut

   for(i=0;i<rangDeb && e != -1;i++)
       e = m->_tas_l[e].prev_in_line; 

   for(i=0; i<nb && e != -1; i++)
   {
      printf("\n i=%ld j=%ld val=%12.6f",m->_tas_l[e].line_index,m->_tas_l[e].col_index, m->_tas_l[e].val);
      e = m->_tas_l[e].prev_in_line;
   }

}
//_____________________________________________________________________________________

/** \brief cree et retourne une copie de la sparse matrix passee en argument**/
//
// on duplique en memoire  directement les deux tableaux _tas_l et _tas_c
// les indices restent identiques
// attention : taille zone a dupliquer en octet
//
struct sparse_matrix_t *sparse_matrix_dupliquer_Tas(struct sparse_matrix_t *m)                                   
{
   struct sparse_matrix_t *copie; 
   
   copie = new_fzsparse_matrix(m->nb_line,m->nb_col, m->nb_item);

   memcpy(copie->line, m->line, sizeof(long int) * m->nb_line);
   memcpy(copie->col, m->col,   sizeof(long int) * m->nb_col);
   memcpy(copie->_tas_l,m->_tas_l,sizeof(struct sparse_item_t) * m->nb_item);
   memcpy(copie->_tas_c,m->_tas_c,sizeof(struct sparse_item_t) * m->nb_item);

   copie->nb_item = m->nb_item;

   return copie;
}

//______________________________________________________________________________
/** \brief Create a sparse matrix **/
// nb_tem correspond au nombre de valeurs non nulles de la matrice
struct sparse_matrix_t *new_fzsparse_matrix(long int nb_line, long int nb_col, long int nb_item)
{
   struct sparse_matrix_t *matrix;
   long int p;

   matrix = (struct sparse_matrix_t *)
      malloc(sizeof(struct sparse_matrix_t));
   assert(matrix);

   matrix->nb_line = nb_line;
   matrix->nb_col = nb_col;
   matrix->line = (long int*)
      calloc(nb_line, sizeof(long int));
   assert(matrix->line);
 
// initialisation a -1
   for(p=0; p < nb_line; p++)
      matrix->line[p]= -1;

   matrix->col = (long int *)
      calloc(nb_col, sizeof(long int));
   assert(matrix->col);

// initialisation a -1
   for(p=0; p < nb_col; p++)
      matrix->col[p]= -1;


   // allocation des deux tas 
   
   matrix->_tas_l  = (struct sparse_item_t *) calloc(nb_item + 1,sizeof(struct sparse_item_t));
   assert(matrix->_tas_l);

   matrix->_tas_c  = (struct sparse_item_t *) calloc(nb_item + 1,sizeof(struct sparse_item_t));
   assert(matrix->_tas_c);

   matrix->nb_item = 0;

   return (matrix);
}
//________________________________________________________________________
// lecture matrice a partir du fichier en utilisant la structure sparse_matrix_t modifiee
//
//structure du ficher :
//
//nombre d'items
//nbre de ligne  ndre_de_colonne
//nol noc valeur
//
//
struct sparse_matrix_t *read_ijk_fzsparse_matrix_Tas(char *filename)
{
   struct sparse_matrix_t *a;
   long int m, n, j, i,nb_item,l;
   long int _index_tas_l;   // indice dans _tas_l
   long int _index_tas_c;   // indice dans _tas_c
   double val;
   struct sparse_item_t *new_item, *v;
   FILE *fd;
   int nb_read;
   long int cpt = 0;

   fprintf(stdout, "reading sparse matrix from '%s' ... ", filename);
   fflush(stdout);

   if (!(fd = fopen(filename, "r"))) {
      perror(filename);
      exit(1);
   }
   nb_read = fscanf(fd, "%ld", &nb_item);
   if (nb_read != 1) {
      fprintf(stdout, "\n");
      fprintf(stderr,
             "read_ijk_sparse_matrix: error reading (nb_item) in '%s'\n",
             filename);
      exit(1);
   }
   nb_read = fscanf(fd, "%ld %ld", &m, &n);
   if (nb_read != 2) {
      fprintf(stdout, "\n");
      fprintf(stderr,
             "read_ijk_sparse_matrix: error reading (m,n) in '%s'\n",
             filename);
      exit(1);
   }
   fprintf(stdout, "(%ldx%ld)-(%ld) ", m, n,nb_item);

   // creation matrice
   a = new_fzsparse_matrix(m, n, nb_item);  

   _index_tas_l = 0;
   
   //============================================================================================
   //  1ere etape : lecture du fichier et remplissage de _tas_l avec chainage /ligne et /colonne
   // 
   while (1) {
   
      nb_read = fscanf(fd, "%ld %ld %lf", &i, &j, &val);
   
      if (feof(fd)) {
         break;
      }
      if (nb_read != 3) {
         fprintf(stdout, "\n");
         fprintf(stderr,
                "read_ijk_sparse_matrix: file '%s' corrupted nread=%d\n",
                filename, nb_read);
         exit(1);
      }
      
      // creation du nouveau maillon et ajout a la matrice
         
      new_item = a->_tas_l + _index_tas_l;  // allocation dans le tas 
      new_item->line_index = i;
      new_item->col_index = j;
      new_item->val = val;
         
      new_item->prev_in_line = a->line[i];  // lien vers l'ancien dernier de la ligne ou -1 si c'est le premier
      new_item->prev_in_col  = a->col[j]; 	// idem
      a->line[i] = _index_tas_l;            
      a->col[j]  = _index_tas_l;
      cpt++;
      _index_tas_l ++;
   }

   a->nb_item = cpt;
   
   //============================================================================================
   //  2eme etape : on construit _tas_c a partir de _tas_l 
   // 
   _index_tas_c = 0;  // indice dans _tab_c de la prochaine case libre

   // on parcourt les colonnes
   printf("\nReorganisation liste des colonnes\n");
   
   for(i=n-1; i >=0; i--)
   {
      // on recopie la liste des elemnts de la colonnes dans _tas_c  en partant de la fin
      l = a->col[i];   // indice case courante dans _tas_l
      a->col[i] = -1;
      
      if(l != -1)  // liste non vide : on recopie 
      {
         while(l != -1)
         {
            new_item = a->_tas_c + _index_tas_c;  // allocation dans le tas 
            new_item->line_index = a->_tas_l[l].line_index;
            new_item->col_index  = a->_tas_l[l].col_index;
            new_item->val = a->_tas_l[l].val;
            new_item->prev_in_col  = a->col[i]; 
            a->col[i] = _index_tas_c;
            l = a->_tas_l[l].prev_in_col;
         
            _index_tas_c++;
         }
      }
      
      
   }

   printf("\nFin lecture matrice");


   return a;
}
//__________________________________________________________________

void free_fzsparse_matrix(struct sparse_matrix_t *m)
{

   fprintf(stdout, "free sparse matrix (%p)\n", m);
   fflush(stdout);

   free(m->line);
   free(m->col);
   free(m->_tas_l);
   free(m->_tas_c);
   free(m);
}
//____________________________________________________________
 
void show_sparse_stats(struct sparse_matrix_t *A)
{
   float density;
 
   density = 100.0 * (double) A->nb_item /
       ((double) A->nb_line * (double) A->nb_col);
 
   fprintf(stdout, "sparse stats (%p):\n", A);
   fprintf(stdout, "\tsize: %ldx%ld, nb items: %ld\n",
           A->nb_line, A->nb_col, A->nb_item);
   fprintf(stdout, "\tdensity: %f\n", density);
}
