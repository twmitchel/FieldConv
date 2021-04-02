#ifndef FILE_TRICKS_INCLUDED
#define FILE_TRICKS_INCLUDED


#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/stat.h>
#include <omp.h>
#include <Misha/Geometry.h>
#include <omp.h>

void GetFilesInDirectory(std::vector<std::string> &out, const std::string &directory);

////////////////////
/// Definitions ////
////////////////////

// Returns a list of files in a directory (except the ones that begin with a dot) 
// From https://stackoverflow.com/a/1932861/10841918
void GetFilesInDirectory(std::vector<std::string> &out, const std::string &directory)
{
#ifdef WINDOWS
    HANDLE dir;
    WIN32_FIND_DATA file_data;

    if ((dir = FindFirstFile((directory + "/*").c_str(), &file_data)) == INVALID_HANDLE_VALUE)
        return; /* No files found */

    do {
        const std::string file_name = file_data.cFileName;
        const std::string full_file_name = directory + "/" + file_name;
        const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

        if (file_name[0] == '.')
            continue;

        if (is_directory)
            continue;

        out.push_back(full_file_name);
    } while (FindNextFile(dir, &file_data));

    FindClose(dir);
#else
    DIR *dir;
    class dirent *ent;
    class stat st;

    dir = opendir(directory.c_str());
    while ((ent = readdir(dir)) != NULL) {
        const std::string file_name = ent->d_name;
        const std::string full_file_name = directory + "/" + file_name;

        if (file_name[0] == '.')
            continue;

        if (stat(full_file_name.c_str(), &st) == -1)
            continue;

        const bool is_directory = (st.st_mode & S_IFDIR) != 0;

        if (is_directory)
            continue;

        out.push_back(full_file_name);
    }
    closedir(dir);
#endif
} // GetFilesInDirectory

template<class Real>
void vectors2File ( std::string fileName, const std::vector<std::vector<Real>>& V)
{

  std::ofstream vF;
  vF.open ( fileName.c_str () );

   for ( int i = 0; i < V.size (); i++)
   {
      int length = (int) V[i].size ();

      for (int j = 0; j < length-1; j++)
      {
         if (std::isnan (V[i][j]) || std::isinf(V[i][j]) )
         {
            vF << 0.0 << " ";
         }
         else
         {
            vF << V[i][j] << " ";
         }
      }

      if (std::isnan (V[i][length-1]) || std::isinf (V[i][length-1]) )
      {
         vF << 0.0 << std::endl;
      }
      else
      {
         vF << V[i][length-1] << std::endl;
      }
   }

   vF.close ();
}

template<class Real>
void vectors2File ( std::string fileName, const std::vector<Point3D<Real>>& V)
{

  std::ofstream vF;
  vF.open ( fileName.c_str () );

   for ( int i = 0; i < V.size (); i++)
   {

      for (int j = 0; j < 3; j++)
      {
         if (std::isnan (V[i][j]) || std::isinf(V[i][j]) )
         {
            vF << 0.0 << " ";
         }
         else
         {
            vF << V[i][j] << " ";
         }
      }

      if (std::isnan (V[i][2]) || std::isinf (V[i][2]) )
      {
         vF << 0.0 << std::endl;
      }
      else
      {
         vF << V[i][2] << std::endl;
      }
   }

   vF.close ();
}


template<class Real>
void file2Vectors ( std::string fileName, std::vector<std::vector<Real>>& V)
{
   #pragma omp critical
   {
      std::string line;

      std::ifstream vF;
      vF.open ( fileName );

      while ( std::getline (vF, line) )
      {
         std::vector<Real> fVec;

         std::stringstream ss (line);

         while (! ss.eof () )
         {
            Real f;
            
            ss >> f;

            if ( std::isnan(f) )
            {
               f = (Real) 0;
            }

            fVec.push_back ( f );
         }

         V.push_back ( fVec);

      }
   }
}

template<class Real>
void vector2File ( std::string fileName, std::vector<Real>& V)
{

   std::ofstream vF;
   vF.open ( fileName.c_str () );

   for ( int i = 0; i < V.size (); i++)
   {
      vF << V[i] << std::endl;
   }
   
   vF.close ();
}

template<class Real>
void file2Vector ( std::string fileName, std::vector<Real>& V)
{
   #pragma omp critical
   {
      std::string line;

      std::ifstream vF;
      vF.open ( fileName );

      while ( std::getline (vF, line) )
      {
         std::vector<Real> fVec;

         std::stringstream ss (line);

         Real f;
            
         ss >> f;

         if ( std::isnan(f) )
         {
            f = (Real) 0;
         }

         V.push_back (f);

      }
   }

}


// copy in binary mode


bool copyFile(const char *SRC, const char* DEST)
{

    std::ifstream src(SRC, std::ios::binary);
    std::ofstream dest(DEST, std::ios::binary);
    dest << src.rdbuf();
    return src && dest;
    
}


bool copyFile(std::string SRC, std::string DEST)
{
    return copyFile ( SRC.c_str (), DEST.c_str () );
}


// Find index
template<class Thing>
int findIndex(std::vector<Thing>& vec, Thing val)
{
   auto look = std::find(vec.begin(), vec.end(), val) - vec.begin();
   
   if (look != vec.end())
   {
      return look - vec.begin();
   }
   else
   {
      return -1;
   }
}



#endif
