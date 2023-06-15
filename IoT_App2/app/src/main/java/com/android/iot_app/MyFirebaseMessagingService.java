    package com.android.iot_app;
    import android.app.Notification;
    import android.app.NotificationChannel;
    import android.app.NotificationManager;
    import android.app.PendingIntent;
    import android.content.Context;
    import android.content.Intent;
    import android.os.Build;
    import android.util.Log;

    import androidx.annotation.NonNull;
    import androidx.core.app.NotificationCompat;
    import androidx.core.app.NotificationManagerCompat;

    import com.google.firebase.messaging.FirebaseMessagingService;
    import com.google.firebase.messaging.RemoteMessage;

    import java.util.Map;

    import static androidx.constraintlayout.widget.ConstraintLayoutStates.TAG;

    public class MyFirebaseMessagingService extends FirebaseMessagingService {

        @Override
        public void onMessageReceived(RemoteMessage remoteMessage) {
            super.onMessageReceived(remoteMessage);

            NotificationManagerCompat notificationManager = NotificationManagerCompat.from(getApplicationContext());


            // 메시지 페이로드 됐는지 확인
            if (remoteMessage.getNotification() != null) {
                String title = remoteMessage.getNotification().getTitle();
                String body = remoteMessage.getNotification().getBody();

                // 인텐트 생성
                Intent intent = new Intent(this, MainActivity.class);
                intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);

                int flags = PendingIntent.FLAG_UPDATE_CURRENT | (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M ? PendingIntent.FLAG_IMMUTABLE : 0);

                PendingIntent pendingIntent = PendingIntent.getActivity(this, 0, intent, flags);

                String channelId = "Default";
                NotificationCompat.Builder builder = new NotificationCompat.Builder(this, channelId)
                        .setSmallIcon(R.mipmap.ic_launcher)
                        .setContentTitle(title)
                        .setContentText(body)
                        .setAutoCancel(true)
                        .setContentIntent(pendingIntent);


                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    String channelName = "Default Channel";
                    NotificationChannel channel = new NotificationChannel(channelId, channelName, NotificationManager.IMPORTANCE_DEFAULT);
//                    NotificationManager notificationManager = getSystemService(NotificationManager.class);
                    notificationManager.createNotificationChannel(channel);
                }

                notificationManager.notify(0, builder.build());
            }
        }
    }