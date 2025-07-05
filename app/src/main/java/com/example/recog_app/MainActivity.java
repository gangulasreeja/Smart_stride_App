package com.example.recog_app;

import android.content.Context;
import android.content.Intent;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;

import com.example.recog_app.R;

import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements SensorEventListener {
    private SensorManager sensorManager;
    private Sensor accelerometer;
    private TextView rawX, rawY, rawZ, filteredX, filteredY, filteredZ, predictionText;
    private KalmanFilter kalmanX, kalmanY, kalmanZ;
    private Interpreter tflite;
    private List<float[]> sensorDataWindow = new ArrayList<>();
    private static final int WINDOW_SIZE = 150;



    // Label mapping (must match Python training order)
    private final String[] activityLabels = {
            "Climbing Down",
            "Climbing Up",
            "Running",
            "Sitting",
            "Walking"
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_prediction);
        // Initialize all UI components
        rawX = findViewById(R.id.rawX);
        rawY = findViewById(R.id.rawY);
        rawZ = findViewById(R.id.rawZ);
        filteredX = findViewById(R.id.filteredX);
        filteredY = findViewById(R.id.filteredY);
        filteredZ = findViewById(R.id.filteredZ);
        predictionText = findViewById(R.id.predictionText);

        // Initialize Kalman filters
        kalmanX = new KalmanFilter();
        kalmanY = new KalmanFilter();
        kalmanZ = new KalmanFilter();

        // Load TFLite model
        try {
            tflite = new Interpreter(loadModelFile(this,"final_2.tflite"));
        } catch (IOException e) {
            Log.e("TFLite", "Error loading model", e);
            Toast.makeText(this, "Failed to load model", Toast.LENGTH_LONG).show();
        }

        // Initialize sensors
        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        if (sensorManager != null) {
            accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
            if (accelerometer == null) {
                accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
                Toast.makeText(this, "Using standard accelerometer", Toast.LENGTH_SHORT).show();
            }
            if (accelerometer == null) {
                Toast.makeText(this, "No accelerometer found!", Toast.LENGTH_LONG).show();
            }
        }

        Button stopButton = findViewById(R.id.stopButton);
        stopButton.setOnClickListener(v -> {
            Intent intent = new Intent(MainActivity.this, HomeActivity.class);
            intent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP); // Clears activity stack
            startActivity(intent);
            finish(); // Close this activity
        });
    }

    // Load TFLite model from assets
    private MappedByteBuffer loadModelFile(Context context,String s) throws IOException {
        String modelPath = "final_2.tflite";
        FileInputStream fileInputStream = context.getAssets().openFd(modelPath).createInputStream();
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffset = context.getAssets().openFd(modelPath).getStartOffset();
        long declaredLength = context.getAssets().openFd(modelPath).getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (accelerometer != null) {
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        sensorManager.unregisterListener(this);
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() != Sensor.TYPE_LINEAR_ACCELERATION &&
                event.sensor.getType() != Sensor.TYPE_ACCELEROMETER) {
            return;
        }

        // Get raw sensor values
        float rawXVal = event.values[0];
        float rawYVal = event.values[1];
        float rawZVal = event.values[2];

        // Update raw values UI
        updateRawUI(rawXVal, rawYVal, rawZVal);

        // Apply Kalman filter
        float x = kalmanX.update(rawXVal);
        float y = kalmanY.update(rawYVal);
        float z = kalmanZ.update(rawZVal);

        // Update filtered values UI
        updateFilteredUI(x, y, z);

        // Add to sliding window
        sensorDataWindow.add(new float[]{x, y, z});

        // Maintain window size
        if (sensorDataWindow.size() > WINDOW_SIZE) {
            sensorDataWindow.remove(0);
        }

        // Run prediction when we have enough data
        if (sensorDataWindow.size() == WINDOW_SIZE && tflite != null) {
            runmodel();
        }
    }

    private void runmodel() {
        // Prepare input buffer (shape: [1, 150, 3])
        float[][][] input = new float[1][WINDOW_SIZE][3];
        for (int i = 0; i < WINDOW_SIZE; i++) {
            input[0][i] = sensorDataWindow.get(i);
        }

        // Prepare output buffer (shape: [1, num_classes])
        float[][] output = new float[1][activityLabels.length];

        // Run model
        tflite.run(input, output);

        // Get predicted class
        int predictedClass = argMax(output[0]);
        String activity = activityLabels[predictedClass];

        // Update UI with prediction
        updatePredictionUI(activity, output[0]);
    }

    private int argMax(float[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private void updateRawUI(float x, float y, float z) {
        runOnUiThread(() -> {
            rawX.setText(String.format("X: %.2f", x));
            rawY.setText(String.format("Y: %.2f", y));
            rawZ.setText(String.format("Z: %.2f", z));
        });
    }

    private void updateFilteredUI(float x, float y, float z) {
        runOnUiThread(() -> {
            filteredX.setText(String.format("filteredX: %.2f", x));
            filteredY.setText(String.format("filteredY: %.2f", y));
            filteredZ.setText(String.format("filteredZ: %.2f", z));
        });
    }

    private void updatePredictionUI(String activity, float[] probabilities) {
        runOnUiThread(() -> {
            // Format probabilities for display
            StringBuilder probText = new StringBuilder();
            for (int i = 0; i < activityLabels.length; i++) {
                probText.append(String.format("%s: %.1f%%\n", activityLabels[i], probabilities[i] * 100));
            }

            predictionText.setText(String.format(
                    "Predicted Activity: %s\n\nConfidence:\n%s",
                    activity,
                    probText.toString()
            ));
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (tflite != null) {
            tflite.close();
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Optional: Handle accuracy changes
    }
}