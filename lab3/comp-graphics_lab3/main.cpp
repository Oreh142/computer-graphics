#include "tgaimage.h"
#include "model.h"
#include "geometry.h"
#include "openfile.h"

Model* model = NULL;
const int width = 800;
const int height = 800;
const int depth = 255;
const Vec3f lightDirection = Vec3f(0, 0, 1);
const Vec3f eyePos = Vec3f(0, 0, 5);
const Vec3f centerPos = Vec3f(0, 0, 0);
const Vec3f upVector = Vec3f(0, 1, 0);
const float shininess = 64.0f;
const float ambientStrength = 0.3f;
const float specularStrength = 0.2f;
const float diffuseStrength = 0.7f;

TGAColor phongShader(const Vec3f& normal_, const Vec3f& fragPos_, const Vec3f& lightDir_, const Vec3f& viewDir_, const TGAColor& color_) {
    Vec3f N = normal_;
    N.normalize();
    Vec3f L = lightDir_;
    L.normalize();
    Vec3f V = viewDir_;
    V.normalize();
    Vec3f C = Vec3f(color_.r / 255.f, color_.g / 255.f, color_.b / 255.f);

    Vec3f ambientComponent = Vec3f(C.x * ambientStrength, C.y * ambientStrength, C.z * ambientStrength);

    float diff = std::max(N * L, 0.f);
    Vec3f diffuseComponent = Vec3f(C.x * diff * diffuseStrength, C.y * diff * diffuseStrength, C.z * diff * diffuseStrength);

    Vec3f reflectDir = (N * 2.f * (N * L) - L).normalize();
    float spec = std::pow(std::max(reflectDir * V, 0.f), shininess);
    Vec3f specularComponent = Vec3f(specularStrength * spec, specularStrength * spec, specularStrength * spec);

    Vec3f result = ambientComponent + diffuseComponent + specularComponent;

    result.x = std::min(255.f, result.x * 255.f);
    result.y = std::min(255.f, result.y * 255.f);
    result.z = std::min(255.f, result.z * 255.f);

    return TGAColor(result.x, result.y, result.z, color_.a);
}

Vec3f getBarycentricCoords(Vec3f P, Vec3f A, Vec3f B, Vec3f C) {
    Vec3f s0(C.x - A.x, B.x - A.x, A.x - P.x);
    Vec3f s1(C.y - A.y, B.y - A.y, A.y - P.y);
    Vec3f u = s0 ^ s1;
    if (std::abs(u.z) < 1) return Vec3f(-1, 1, 1);
    return Vec3f(1.f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
}

void drawTriangle(Vec3f t0, Vec3f t1, Vec3f t2, Vec2f uv0, Vec2f uv1, Vec2f uv2, Vec3f n0, Vec3f n1, Vec3f n2, TGAImage& image, TGAImage& texture_image, int* zbuffer) {
    int minX = std::max(0, (int)std::min({ t0.x, t1.x, t2.x }));
    int maxX = std::min(width - 1, (int)std::max({ t0.x, t1.x, t2.x }));
    int minY = std::max(0, (int)std::min({ t0.y, t1.y, t2.y }));
    int maxY = std::min(height - 1, (int)std::max({ t0.y, t1.y, t2.y }));

    for (int x = minX; x <= maxX; x++) {
        for (int y = minY; y <= maxY; y++) {
            Vec3f baryCoord = getBarycentricCoords(Vec3f(x, y, 0), t0, t1, t2);
            if (baryCoord.x < 0 || baryCoord.y < 0 || baryCoord.z < 0) continue;
            float z = t0.z * baryCoord.x + t1.z * baryCoord.y + t2.z * baryCoord.z;
            int idx = x + y * width;
            if (zbuffer[idx] > z) continue;
            zbuffer[idx] = z;

            Vec3f normal = n0 * baryCoord.x + n1 * baryCoord.y + n2 * baryCoord.z;
            normal.normalize();

            Vec2f uv = uv0 * baryCoord.x + uv1 * baryCoord.y + uv2 * baryCoord.z;
            uv.x *= texture_image.get_width();
            uv.y *= texture_image.get_height();
            TGAColor texColor = texture_image.get(static_cast<int>(uv.x), static_cast<int>(uv.y));

            Vec3f viewDir = (eyePos - (t0 * baryCoord.x + t1 * baryCoord.y + t2 * baryCoord.z)).normalize();

            TGAColor finalColor = phongShader(normal, Vec3f(x, y, 0), lightDirection, viewDir, texColor);
            image.set(x, y, finalColor);
        }
    }
}

Matrix vectorToMatrix(Vec3f v) {
    Matrix result(4, 1);
    result[0][0] = v.x;
    result[1][0] = v.y;
    result[2][0] = v.z;
    result[3][0] = 1.f;
    return result;
}

Vec3f matrixToVector(Matrix m) {
    Vec3f result;
    float w = m[3][0];
    result.x = m[0][0] / w;
    result.y = m[1][0] / w;
    result.z = m[2][0] / w;
    return result;
}

Matrix getCameraViewport(int x, int y, int width, int height, int depth) {
    Matrix result = Matrix::identity(4);
    result[0][0] = width / 2.f;
    result[1][1] = height / 2.f;
    result[2][2] = depth / 2.f;

    result[0][3] = width / 2.f + x;
    result[1][3] = height / 2.f + y;
    result[2][3] = depth / 2.f;
    return result;
}

Matrix getLookAt(Vec3f eye, Vec3f center, Vec3f up) {
    Vec3f z = (eye - center).normalize();
    Vec3f x = (up ^ z).normalize();
    Vec3f y = (z ^ x).normalize();
    Matrix res = Matrix::identity(4);
    for (int i = 0; i < 3; i++) {
        res[0][i] = x[i];
        res[1][i] = y[i];
        res[2][i] = z[i];
        res[i][3] = -center[i];
    }
    return res;
}

int main(int argc, char** argv) {
    TGAImage image(width, height, TGAImage::RGB);
    TGAImage texture;
    texture.read_tga_file("resources/african_head_diffuse.tga");
    texture.flip_vertically();

    model = new Model("resources/african_head.obj");
    int* zbuffer = new int[width * height];
    for (int i = 0; i < width * height; i++) {
        zbuffer[i] = -1e30f;
    }

    Matrix cameraViewport = getCameraViewport(0, 0, width, height, depth);
    Matrix projectionMatrix = Matrix::identity(4);
    Matrix viewModel = getLookAt(eyePos, centerPos, upVector);
    projectionMatrix[3][2] = -1.f / (eyePos - centerPos).norm();

    for (int i = 0; i < model->getNumFaces(); i++) {
        std::vector<int> face = model->getFaceByIndex(i);
        std::vector<int> textureIndices = model->getTextureByIndex(i);
        std::vector<int> normalIndices = model->getNormalByIndex(i);
        Vec3i screenCoords[3];
        Vec3f worldCoords[3];
        Vec2f uvCoords[3];
        Vec3f normalCoords[3];
        for (int j = 0; j < 3; j++) {
            Vec3f vert = model->getVertexByIndex(face[j]);
            screenCoords[j] = matrixToVector(cameraViewport * projectionMatrix * viewModel * vectorToMatrix(vert));
            worldCoords[j] = vert;
            uvCoords[j] = model->getTextureVertexByIndex(textureIndices[j]);
            normalCoords[j] = model->getNormalVertexByIndex(normalIndices[j]);
        }
        drawTriangle(screenCoords[0], screenCoords[1], screenCoords[2], uvCoords[0], uvCoords[1], uvCoords[2], normalCoords[0], normalCoords[1], normalCoords[2], image, texture, zbuffer);
    }

    image.flip_vertically();
    image.write_tga_file("output.tga");
    openFile("output.tga");

    delete[] zbuffer;
    delete model;
}