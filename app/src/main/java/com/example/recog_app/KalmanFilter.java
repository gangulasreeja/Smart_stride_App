package com.example.recog_app;

public class KalmanFilter {
    private float Q = 0.0001f;//process noise
    private float R = 0.01f;//error in measurement tht is noise in measurement
    private float P = 1.0f;//error in estimate
    private float X = 0.0f;//estimated value

    public float update(float measurement) {
        P = P + Q;
        float K = P / (P + R);
        X = X + K * (measurement - X);
        P = (1 - K) * P;
        return X;
    }
}