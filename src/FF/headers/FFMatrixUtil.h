#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace pdb {
class PDBClient;
class CatalogClient;
class String;
} // namespace pdb

namespace ff {
void load_matrix_data(pdb::PDBClient &pdbClient, std::string path,
                      pdb::String dbName, pdb::String setName, int blockX,
                      int blockY, bool dont_pad_x, bool dont_pad_y,
                      std::string &errMsg, int size = 128, bool partitionByCol = true);

void load_matrix_data(pdb::PDBClient &pdbClient, std::string path,
                      pdb::String dbName, pdb::String setName, int pad_x,
                      int pad_y, std::string &errMsg);

void loadMatrix(pdb::PDBClient &pdbClient, pdb::String dbName,
                pdb::String setName, int totalX, int totalY, int blockX,
                int blockY, bool dont_pad_x, bool dont_pad_y,
                std::string &errMsg, int size = 128,
                bool partitionByCol = true);

template <class T>
void loadMatrixGeneric(pdb::PDBClient &pdbClient, pdb::String dbName,
                       pdb::String setName, int totalX, int totalY, int blockX,
                       int blockY, bool dont_pad_x, bool dont_pad_y, std::string &errMsg,
                       int size = 128, bool partitionByCol = true);

template <class T>
void loadMatrixGenericFromFile(pdb::PDBClient &pdbClient, std::string path, pdb::String dbName,
                               pdb::String setName, int totalX, int totalY, int blockX,
                               int blockY, int label_col_index, std::string &errMsg,
                               int size = 128, int numPartitions = 1, bool partitionByCol = true);

void loadMapFromSVMFile(pdb::PDBClient &pdbClient, std::string path,
                        pdb::String dbName, pdb::String setName,
                        int totalX, int totalY,
                        std::string &errMsg, int size = 128, int numPartitions = 1, bool partitionByCol = true);
void loadMapBlockFromSVMFile(pdb::PDBClient &pdbClient, std::string path,
                             pdb::String dbName, pdb::String setName,
                             int totalX, int totalY, int blockXSize,
                             std::string &errMsg, int size = 128, int numPartitions = 1);
void load_matrix_from_file(std::string path,
                           std::vector<std::vector<float>> &matrix);

void print_stats(pdb::PDBClient &pdbClient, std::string dbName,
                 std::string setName);

void print(pdb::PDBClient &pdbClient, std::string dbName, std::string setName);

bool is_empty_set(pdb::PDBClient &pdbClient, pdb::CatalogClient &catalogClient,
                  std::string dbName, std::string setName);
} // namespace ff
