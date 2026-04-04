#include "FreeCamera.h"

using namespace DirectX;

FreeCamera::FreeCamera()
{
    SetLens(0.25f * MathHelper::Pi, 1.0f, 0.1f, 200.0f);
}

void FreeCamera::SetLens(float fovY, float aspect, float zn, float zf)
{
    XMMATRIX proj = XMMatrixPerspectiveFovLH(fovY, aspect, zn, zf);
    XMStoreFloat4x4(&mProj, proj);
}

void FreeCamera::SetPosition(float x, float y, float z)
{
    mPosition = XMFLOAT3(x, y, z);
    mViewDirty = true;
}

void FreeCamera::Walk(float d)
{
    XMVECTOR scale = XMVectorReplicate(d);
    XMVECTOR look = XMLoadFloat3(&mLook);
    XMVECTOR pos = XMLoadFloat3(&mPosition);
    XMStoreFloat3(&mPosition, XMVectorMultiplyAdd(scale, look, pos));
    mViewDirty = true;
}

void FreeCamera::Strafe(float d)
{
    XMVECTOR scale = XMVectorReplicate(d);
    XMVECTOR right = XMLoadFloat3(&mRight);
    XMVECTOR pos = XMLoadFloat3(&mPosition);
    XMStoreFloat3(&mPosition, XMVectorMultiplyAdd(scale, right, pos));
    mViewDirty = true;
}

void FreeCamera::Rise(float d)
{
    mPosition.y += d;
    mViewDirty = true;
}

void FreeCamera::Pitch(float angle)
{
    XMMATRIX rot = XMMatrixRotationAxis(XMLoadFloat3(&mRight), angle);
    XMStoreFloat3(&mUp, XMVector3TransformNormal(XMLoadFloat3(&mUp), rot));
    XMStoreFloat3(&mLook, XMVector3TransformNormal(XMLoadFloat3(&mLook), rot));
    mViewDirty = true;
}

void FreeCamera::Yaw(float angle)
{
    XMMATRIX rot = XMMatrixRotationY(angle);
    XMStoreFloat3(&mRight, XMVector3TransformNormal(XMLoadFloat3(&mRight), rot));
    XMStoreFloat3(&mUp, XMVector3TransformNormal(XMLoadFloat3(&mUp), rot));
    XMStoreFloat3(&mLook, XMVector3TransformNormal(XMLoadFloat3(&mLook), rot));
    mViewDirty = true;
}

void FreeCamera::UpdateViewMatrix()
{
    if (!mViewDirty)
        return;

    XMVECTOR right = XMLoadFloat3(&mRight);
    XMVECTOR up = XMLoadFloat3(&mUp);
    XMVECTOR look = XMLoadFloat3(&mLook);
    XMVECTOR pos = XMLoadFloat3(&mPosition);

    look = XMVector3Normalize(look);
    up = XMVector3Normalize(XMVector3Cross(look, right));
    right = XMVector3Cross(up, look);

    float x = -XMVectorGetX(XMVector3Dot(pos, right));
    float y = -XMVectorGetX(XMVector3Dot(pos, up));
    float z = -XMVectorGetX(XMVector3Dot(pos, look));

    XMStoreFloat3(&mRight, right);
    XMStoreFloat3(&mUp, up);
    XMStoreFloat3(&mLook, look);

    mView(0, 0) = mRight.x;
    mView(1, 0) = mRight.y;
    mView(2, 0) = mRight.z;
    mView(3, 0) = x;

    mView(0, 1) = mUp.x;
    mView(1, 1) = mUp.y;
    mView(2, 1) = mUp.z;
    mView(3, 1) = y;

    mView(0, 2) = mLook.x;
    mView(1, 2) = mLook.y;
    mView(2, 2) = mLook.z;
    mView(3, 2) = z;

    mView(0, 3) = 0.0f;
    mView(1, 3) = 0.0f;
    mView(2, 3) = 0.0f;
    mView(3, 3) = 1.0f;

    mViewDirty = false;
}

XMMATRIX FreeCamera::GetView() const
{
    return XMLoadFloat4x4(&mView);
}

XMMATRIX FreeCamera::GetProj() const
{
    return XMLoadFloat4x4(&mProj);
}

XMFLOAT3 FreeCamera::GetPosition3f() const
{
    return mPosition;
}
