#ifndef __MODEL_H__
#define __MODEL_H__

#include <vector>
#include "geometry.h"

class Model {
private:
    std::vector<Vec3f> vertexes_;
    std::vector<Vec2f> textureVertexes_;
    std::vector<Vec3f> normalVertexes_;
    std::vector<std::vector<int>> faces_;
    std::vector<std::vector<int>> textures_;
    std::vector<std::vector<int>> normals_;
public:
    Model(const char* filename);
    ~Model();
    int getNumVertexes();
    int getNumFaces();
    Vec3f getVertexByIndex(int i);
    Vec2f getTextureVertexByIndex(int i);
    Vec3f getNormalVertexByIndex(int i);
    std::vector<int> getTextureByIndex(int idx);
    std::vector<int> getFaceByIndex(int idx);
    std::vector<int> getNormalByIndex(int idx);
};

#endif //__MODEL_H__