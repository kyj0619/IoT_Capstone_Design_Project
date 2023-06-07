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

import android.animation.ObjectAnimator;
import android.app.PendingIntent;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.animation.DecelerateInterpolator;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;


public class MainActivity extends AppCompatActivity {

//         Database DATA 불러오기
    private TextView textView2;
    private TextView textView4;
    private ImageView imageView;
    private ImageView imageView2;


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
        imageView = (ImageView) findViewById(R.id.imageView);
        imageView2 = (ImageView) findViewById(R.id.imageView2);

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
                        mDatabase.child("phone tokens").setValue(token);
//                        mDatabase.child("wearable tokens").setValue(token); //어플 만질 때는 위에 주석하고 이거 사용
                    }
                });
    }


    @Override
    protected void onStart() {
        super.onStart();

        Intent intent = new Intent(this, FirebaseDatabaseService.class);
        startService(intent);

        sound_type.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                String sound_type = dataSnapshot.getValue(String.class);
                textView2.setText(sound_type);

                if (sound_type.equals("car_siren")) {
                    textView2.setTextColor(Color.RED);
                    imageView2.setImageResource(R.drawable.siren);
                } else if (sound_type.equals("fire_alarm_sound")) {
                    textView2.setTextColor(Color.RED);
                    imageView2.setImageResource(R.drawable.fire);
                } else if (sound_type.equals("car_horn")) {
                    textView2.setTextColor(Color.RED);
                    imageView2.setImageResource(R.drawable.car_horn);
                } else if (sound_type.equals("motorcycle_horn")) {
                    textView2.setTextColor(Color.RED);
                    imageView2.setImageResource(R.drawable.motorcycle_horn);
                } else if (sound_type.equals("name")) {
                    textView2.setTextColor(Color.CYAN);
                    imageView2.setImageResource(R.drawable.name);
                } else if (sound_type.equals("car_drivesound")) {
                    textView2.setTextColor(Color.BLUE);
                    imageView2.setImageResource(R.drawable.car);
                } else if (sound_type.equals("motorcycle_drivesound")) {
                    textView2.setTextColor(Color.BLUE);
                    imageView2.setImageResource(R.drawable.motorcycle);
                } else if (sound_type.equals("bicycle")) {
                    textView2.setTextColor(Color.BLUE);
                    imageView2.setImageResource(R.drawable.bike);
                }
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

                if (sound_direction.equals("north")) {
                    imageView.setImageResource(R.drawable.north);
                } else if (sound_direction.equals("east")) {
                    imageView.setImageResource(R.drawable.east);
                } else if (sound_direction.equals("south")) {
                    imageView.setImageResource(R.drawable.south);
                } else if (sound_direction.equals("west")) {
                    imageView.setImageResource(R.drawable.west);
                }
            }
            @Override
            public void onCancelled(DatabaseError databaseError) {
            }
        });

    }
}



