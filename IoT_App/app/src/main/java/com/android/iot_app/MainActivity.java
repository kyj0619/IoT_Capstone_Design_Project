package com.android.iot_app;

import static androidx.constraintlayout.widget.ConstraintLayoutStates.TAG;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;
import com.google.firebase.messaging.FirebaseMessaging;

import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;



public class MainActivity extends AppCompatActivity {

//         Database DATA 불러오기

    private TextView textView2;
    private TextView textView4;



    DatabaseReference mRootRef = FirebaseDatabase.getInstance().getReference();
    DatabaseReference sound_type = mRootRef.child("DATA/sound_type");
    DatabaseReference sound_detection = mRootRef.child("DATA/sound_direction");

    DatabaseReference mDatabase = FirebaseDatabase.getInstance().getReference();


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        textView2 = (TextView) findViewById(R.id.textView2);
        textView4 = (TextView) findViewById(R.id.textView4);

        FirebaseMessaging.getInstance().getToken()
                .addOnCompleteListener(new OnCompleteListener<String>() {
                    @Override
                    public void onComplete(@NonNull Task<String> task) {
                        if (!task.isSuccessful()) {
                            Log.w(TAG, "Fetching FCM registration token failed", task.getException());
                            return;
                        }

                        // Get new FCM registration token
                        String token = task.getResult();

                        // Log and toast
//                        String msg = getString(R.string.msg_token_fmt, token);
                        Log.d(TAG, "token value " + token);
                        Toast.makeText(MainActivity.this, token, Toast.LENGTH_SHORT).show();
                        mDatabase.child("tokens").setValue(token);
                    }
                });
    }



    @Override
    protected void onStart() {
        super.onStart();

        sound_type.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                String sound_type = dataSnapshot.getValue(String.class);
                textView2.setText(String.valueOf(sound_type));
            }

            @Override
            public void onCancelled(DatabaseError databaseError) {

            }
        });
        sound_detection.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                String sound_direction = dataSnapshot.getValue(String.class);
                textView4.setText(sound_direction);
            }

            @Override
            public void onCancelled(DatabaseError databaseError) {

            }
        });

    }
}



