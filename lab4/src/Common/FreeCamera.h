#pragma once

#include "d3dUtil.h"

class FreeCamera
{
public:
    FreeCamera();

    void SetLens(float fovY, float aspect, float zn, float zf);
    void SetPosition(float x, float y, float z);

    void Walk(float d);
    void Strafe(float d);
    void Rise(float d);

    void Pitch(float angle);
    void Yaw(float angle);

    void UpdateViewMatrix();

    DirectX::XMMATRIX GetView() const;
    DirectX::XMMATRIX GetProj() const;
    DirectX::XMFLOAT3 GetPosition3f() const;

private:
    DirectX::XMFLOAT3 mPosition = { 0.0f, 0.0f, 0.0f };
    DirectX::XMFLOAT3 mRight = { 1.0f, 0.0f, 0.0f };
    DirectX::XMFLOAT3 mUp = { 0.0f, 1.0f, 0.0f };
    DirectX::XMFLOAT3 mLook = { 0.0f, 0.0f, 1.0f };

    DirectX::XMFLOAT4X4 mView = MathHelper::Identity4x4();
    DirectX::XMFLOAT4X4 mProj = MathHelper::Identity4x4();

    bool mViewDirty = true;
};
