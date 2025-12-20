#include "tgaimage.h"
#include "model.h"
#include "geometry.h"
#include "openfile.h"

#include <limits>
#include <algorithm>
#include <vector>
#include <cmath>

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

// --------- helpers: clamp without C++17 ----------
static inline int clampi(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}
static inline unsigned char clamp_u8(int v) {
    return (unsigned char)clampi(v, 0, 255);
}

// --------- phong ----------
TGAColor phongShader(const Vec3f& normal_, const Vec3f& fragPos_, const Vec3f& lightDir_, const Vec3f& viewDir_, const TGAColor& color_) {
    Vec3f N = normal_; N.normalize();
    Vec3f L = lightDir_; L.normalize();
    Vec3f V = viewDir_; V.normalize();
    Vec3f C = Vec3f(color_.r / 255.f, color_.g / 255.f, color_.b / 255.f);

    Vec3f ambientComponent(C.x * ambientStrength, C.y * ambientStrength, C.z * ambientStrength);

    float diff = std::max(N * L, 0.f);
    Vec3f diffuseComponent(C.x * diff * diffuseStrength, C.y * diff * diffuseStrength, C.z * diff * diffuseStrength);

    Vec3f reflectDir = (N * 2.f * (N * L) - L).normalize();
    float spec = std::pow(std::max(reflectDir * V, 0.f), shininess);
    Vec3f specularComponent(specularStrength * spec, specularStrength * spec, specularStrength * spec);

    Vec3f result = ambientComponent + diffuseComponent + specularComponent;

    result.x = std::min(255.f, result.x * 255.f);
    result.y = std::min(255.f, result.y * 255.f);
    result.z = std::min(255.f, result.z * 255.f);

    return TGAColor((unsigned char)result.x, (unsigned char)result.y, (unsigned char)result.z, color_.a);
}

// --------- barycentric ----------
Vec3f getBarycentricCoords(Vec3f P, Vec3f A, Vec3f B, Vec3f C) {
    Vec3f s0(C.x - A.x, B.x - A.x, A.x - P.x);
    Vec3f s1(C.y - A.y, B.y - A.y, A.y - P.y);
    Vec3f u = s0 ^ s1;
    if (std::abs(u.z) < 1) return Vec3f(-1, 1, 1);
    return Vec3f(1.f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
}

// --------- textured triangle ----------
void drawTriangle(Vec3f t0, Vec3f t1, Vec3f t2,
    Vec2f uv0, Vec2f uv1, Vec2f uv2,
    Vec3f n0, Vec3f n1, Vec3f n2,
    TGAImage& image, TGAImage& texture_image, float* zbuffer) {

    int minX = std::max(0, (int)std::min({ t0.x, t1.x, t2.x }));
    int maxX = std::min(width - 1, (int)std::max({ t0.x, t1.x, t2.x }));
    int minY = std::max(0, (int)std::min({ t0.y, t1.y, t2.y }));
    int maxY = std::min(height - 1, (int)std::max({ t0.y, t1.y, t2.y }));

    for (int x = minX; x <= maxX; x++) {
        for (int y = minY; y <= maxY; y++) {
            Vec3f baryCoord = getBarycentricCoords(Vec3f((float)x, (float)y, 0), t0, t1, t2);
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

            int tx = clampi((int)uv.x, 0, texture_image.get_width() - 1);
            int ty = clampi((int)uv.y, 0, texture_image.get_height() - 1);

            // ВАЖНО: texture_image НЕ const, иначе get() не вызовется
            TGAColor texColor = texture_image.get(tx, ty);

            Vec3f viewDir = (eyePos - (t0 * baryCoord.x + t1 * baryCoord.y + t2 * baryCoord.z)).normalize();
            TGAColor finalColor = phongShader(normal, Vec3f((float)x, (float)y, 0), lightDirection, viewDir, texColor);

            image.set(x, y, finalColor);
        }
    }
}

// --------- solid triangle into RGBA overlay ----------
void drawTriangleSolid(Vec3f t0, Vec3f t1, Vec3f t2, TGAImage& overlayRGBA, float* zbuffer, const TGAColor& colorRGBA) {
    int minX = std::max(0, (int)std::min({ t0.x, t1.x, t2.x }));
    int maxX = std::min(width - 1, (int)std::max({ t0.x, t1.x, t2.x }));
    int minY = std::max(0, (int)std::min({ t0.y, t1.y, t2.y }));
    int maxY = std::min(height - 1, (int)std::max({ t0.y, t1.y, t2.y }));

    for (int x = minX; x <= maxX; x++) {
        for (int y = minY; y <= maxY; y++) {
            Vec3f bc = getBarycentricCoords(Vec3f((float)x, (float)y, 0), t0, t1, t2);
            if (bc.x < 0 || bc.y < 0 || bc.z < 0) continue;

            float z = t0.z * bc.x + t1.z * bc.y + t2.z * bc.z;
            int idx = x + y * width;
            if (zbuffer[idx] > z) continue;
            zbuffer[idx] = z;

            overlayRGBA.set(x, y, colorRGBA);
        }
    }
}

// --------- alpha blend overlay over base ----------
// overlayRGBA НЕ const, потому что TGAImage::get() у тебя не const-метод
void alphaBlendOver(TGAImage& base, TGAImage& overlayRGBA) {
    for (int y = 0; y < base.get_height(); y++) {
        for (int x = 0; x < base.get_width(); x++) {
            TGAColor s = overlayRGBA.get(x, y);
            if (s.a == 0) continue;

            TGAColor d = base.get(x, y);
            float a = s.a / 255.f;

            int outR = (int)std::round(d.r * (1.f - a) + s.r * a);
            int outG = (int)std::round(d.g * (1.f - a) + s.g * a);
            int outB = (int)std::round(d.b * (1.f - a) + s.b * a);

            base.set(x, y, TGAColor(clamp_u8(outR), clamp_u8(outG), clamp_u8(outB), 255));
        }
    }
}

// --------- matrices ----------
Matrix vectorToMatrix(Vec3f v) {
    Matrix result(4, 1);
    result[0][0] = v.x;
    result[1][0] = v.y;
    result[2][0] = v.z;
    result[3][0] = 1.f;
    return result;
}

Vec3f matrixToVector(Matrix m) {
    float w = m[3][0];
    return Vec3f(m[0][0] / w, m[1][0] / w, m[2][0] / w);
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

    // --- основной zbuffer ---
    float* zbuffer = new float[width * height];
    for (int i = 0; i < width * height; i++) {
        zbuffer[i] = -std::numeric_limits<float>::infinity();
    }

    Matrix cameraViewport = getCameraViewport(0, 0, width, height, depth);
    Matrix projectionMatrix = Matrix::identity(4);
    Matrix viewModel = getLookAt(eyePos, centerPos, upVector);
    projectionMatrix[3][2] = -1.f / (eyePos - centerPos).norm();

    // --- рендер модели ---
    for (int i = 0; i < model->getNumFaces(); i++) {
        std::vector<int> face = model->getFaceByIndex(i);
        std::vector<int> textureIndices = model->getTextureByIndex(i);
        std::vector<int> normalIndices = model->getNormalByIndex(i);

        Vec3f screenCoords[3];
        Vec2f uvCoords[3];
        Vec3f normalCoords[3];

        for (int j = 0; j < 3; j++) {
            Vec3f vert = model->getVertexByIndex(face[j]);
            screenCoords[j] = matrixToVector(cameraViewport * projectionMatrix * viewModel * vectorToMatrix(vert));
            uvCoords[j] = model->getTextureVertexByIndex(textureIndices[j]);
            normalCoords[j] = model->getNormalVertexByIndex(normalIndices[j]);
        }

        drawTriangle(screenCoords[0], screenCoords[1], screenCoords[2],
            uvCoords[0], uvCoords[1], uvCoords[2],
            normalCoords[0], normalCoords[1], normalCoords[2],
            image, texture, zbuffer);
    }

    // =======================
    //  ПОЛУПРОЗРАЧНЫЙ КУБ AABB
    // =======================

    // AABB по вершинам
    Vec3f bmin(std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity());
    Vec3f bmax(-std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity());

    for (int i = 0; i < model->getNumVertexes(); i++) {
        Vec3f v = model->getVertexByIndex(i);
        bmin.x = std::min(bmin.x, v.x); bmin.y = std::min(bmin.y, v.y); bmin.z = std::min(bmin.z, v.z);
        bmax.x = std::max(bmax.x, v.x); bmax.y = std::max(bmax.y, v.y); bmax.z = std::max(bmax.z, v.z);
    }

    Vec3f size = bmax - bmin;
    float pad = 0.05f * std::max(std::max(std::abs(size.x), std::abs(size.y)), std::abs(size.z));
    bmin = bmin - Vec3f(pad, pad, pad);
    bmax = bmax + Vec3f(pad, pad, pad);

    // 8 вершин куба
    Vec3f cv[8] = {
        {bmin.x, bmin.y, bmin.z}, // 0
        {bmax.x, bmin.y, bmin.z}, // 1
        {bmax.x, bmax.y, bmin.z}, // 2
        {bmin.x, bmax.y, bmin.z}, // 3
        {bmin.x, bmin.y, bmax.z}, // 4
        {bmax.x, bmin.y, bmax.z}, // 5
        {bmax.x, bmax.y, bmax.z}, // 6
        {bmin.x, bmax.y, bmax.z}, // 7
    };

    int tris[12][3] = {
        {0,1,2}, {0,2,3}, // z=min
        {4,6,5}, {4,7,6}, // z=max
        {0,4,5}, {0,5,1}, // y=min
        {3,2,6}, {3,6,7}, // y=max
        {0,3,7}, {0,7,4}, // x=min
        {1,5,6}, {1,6,2}  // x=max
    };

    // overlay RGBA + свой zbuffer
    TGAImage cubeOverlay(width, height, TGAImage::RGBA);
    float* cubeZ = new float[width * height];
    for (int i = 0; i < width * height; i++) cubeZ[i] = -std::numeric_limits<float>::infinity();

    // Полупрозрачный цвет
    TGAColor cubeColor(60, 200, 255, 90);

    for (int i = 0; i < 12; i++) {
        Vec3f w0 = cv[tris[i][0]];
        Vec3f w1 = cv[tris[i][1]];
        Vec3f w2 = cv[tris[i][2]];

        Vec3f s0 = matrixToVector(cameraViewport * projectionMatrix * viewModel * vectorToMatrix(w0));
        Vec3f s1 = matrixToVector(cameraViewport * projectionMatrix * viewModel * vectorToMatrix(w1));
        Vec3f s2 = matrixToVector(cameraViewport * projectionMatrix * viewModel * vectorToMatrix(w2));

        drawTriangleSolid(s0, s1, s2, cubeOverlay, cubeZ, cubeColor);
    }

    // Наложить куб на картинку
    alphaBlendOver(image, cubeOverlay);

    // =======================

    image.flip_vertically();
    image.write_tga_file("output.tga");
    openFile("output.tga");

    delete[] cubeZ;
    delete[] zbuffer;
    delete model;
    return 0;
}
