package com.android.iot_app;

import static androidx.constraintlayout.widget.ConstraintLayoutStates.TAG;

import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.os.IBinder;
import android.os.Vibrator;
import android.util.Log;

import androidx.annotation.Nullable;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

public class FirebaseDatabaseService extends Service {

    private DatabaseReference sound_detection;

    @Override
    public void onCreate() {
        super.onCreate();
        sound_detection = FirebaseDatabase.getInstance().getReference("DATA/sound_direction");
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        sound_detection.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                String sound_direction = dataSnapshot.getValue(String.class);

                // Get an instance of the Vibrator
                Vibrator vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);

                // Different vibration patterns based on the sound direction
                long[] eastVibrationPattern = {0, 200};
                long[] westVibrationPattern = {0, 500};
                long[] southVibrationPattern = {0, 200, 250, 200};
                long[] northVibrationPattern = {0, 500, 250, 500};

                // Change the vibration pattern based on the sound direction
                switch (sound_direction) {
                    case "north":
                        vibrator.vibrate(northVibrationPattern, -1);
                        break;
                    case "west":
                        vibrator.vibrate(westVibrationPattern, -1);
                        break;
                    case "east":
                        vibrator.vibrate(eastVibrationPattern, -1);
                        break;
                    case "south":
                        vibrator.vibrate(southVibrationPattern, -1);
                        break;
                    default:
                        break;
                }
            }

            @Override
            public void onCancelled(DatabaseError databaseError) {
                Log.e(TAG, "onCancelled: ", databaseError.toException());
            }
        });

        return START_STICKY;
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
