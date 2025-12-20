#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include "model.h"

Model::Model(const char* filename) : vertexes_(), faces_() {
    std::ifstream in;
    in.open(filename, std::ifstream::in);
    if (in.fail()) return;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line.c_str());
        char trash;
        if (!line.compare(0, 2, "v ")) {
            iss >> trash;
            Vec3f v;
            for (int i = 0; i < 3; i++) iss >> v[i];
            vertexes_.push_back(v);
        }
        else if (!line.compare(0, 2, "f ")) {
            iss >> trash;
            std::vector<int> face;
            std::vector<int> texture;
            std::vector<int> normal;
            int fidx;
            int vertex_index, texture_index, normal_index;
            char separator;

            while (iss >> vertex_index) {
                face.push_back(vertex_index - 1);
                if (iss.peek() == '/') {
                    iss >> separator;
                    if (iss.peek() != '/') {
                        iss >> texture_index;
                        texture.push_back(texture_index - 1);
                    }
                    if (iss.peek() == '/') {
                        iss >> separator;
                        iss >> normal_index;
                        normal.push_back(normal_index - 1);
                    }
                    iss >> std::ws;
                }
            }

            faces_.push_back(std::move(face));
            textures_.push_back(std::move(texture));
            normals_.push_back(std::move(normal));
        }
        else if (!line.compare(0, 3, "vt ")) {
            iss >> trash >> trash;
            Vec2f vt;
            for (int i = 0; i < 2; i++) iss >> vt[i];
            textureVertexes_.push_back(vt);
        }
        else if (!line.compare(0, 3, "vn ")) {
            iss >> trash >> trash;
            Vec3f vn;
            for (int i = 0; i < 3; i++) iss >> vn[i];
            normalVertexes_.push_back(vn);
        }
    }
    std::cerr << "Vertex number: " << vertexes_.size() << ". Faces number: " << faces_.size() << ". Textures number: " << textures_.size() << ". Normals number: " << normals_.size() << std::endl;
}

Model::~Model() {
}

int Model::getNumVertexes() {
    return (int)vertexes_.size();
}

int Model::getNumFaces() {
    return (int)faces_.size();
}

std::vector<int> Model::getFaceByIndex(int idx) {
    return faces_[idx];
}

Vec3f Model::getVertexByIndex(int i) {
    return vertexes_[i];
}

std::vector<int> Model::getTextureByIndex(int i) {
    return textures_[i];
}

Vec2f Model::getTextureVertexByIndex(int i) {
    return textureVertexes_[i];
}

Vec3f Model::getNormalVertexByIndex(int i) {
    return normalVertexes_[i];
}

std::vector<int> Model::getNormalByIndex(int i) {
    return normals_[i];
}