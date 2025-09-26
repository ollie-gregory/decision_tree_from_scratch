#ifndef CSVReader
#define CSVReader

#include <vector>
#include <string>

// Function to parse a single CSV row into fields
std::vector<std::string> parseCSVRow(const std::string& row);

// Function to read an entire CSV file
std::vector<std::vector<std::string>> readCSV(const std::string& filename);

#endif