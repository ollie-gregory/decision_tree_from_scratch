#include <iostream>
#include <fstream>
#include <sstream>

std::vector<std::string> parseCSVRow(const std::string& row) {
    std::vector<std::string> fields;
    std::string field;
    bool inQuotes = false;
    
    for (char c : row) {
        if (!inQuotes && c == ',') {
            fields.push_back(field);
            field.clear();
        } else if (c == '"') {
            inQuotes = !inQuotes;
        } else {
            field += c;
        }
    }
    fields.push_back(field);
    
    return fields;
}

std::vector<std::vector<std::string>> readCSV(const std::string& filename) {
    std::vector<std::vector<std::string>> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        data.push_back(parseCSVRow(line));
    }

    file.close();
    return data;
}