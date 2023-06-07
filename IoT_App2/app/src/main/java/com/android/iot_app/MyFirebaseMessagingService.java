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

//        test message recieved 테스트 메시지 수신
//        @Override
//        public void onMessageReceived(@NonNull RemoteMessage remoteMessage) {
//            super.onMessageReceived(remoteMessage);
//
//            NotificationManagerCompat notificationManager = NotificationManagerCompat.from(getApplicationContext());
//
//            NotificationCompat.Builder builder = null;
//            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
//                if (notificationManager.getNotificationChannel("MyNotificationChannelID") == null) {
//                    NotificationChannel channel = new NotificationChannel("MyNotificationChannelName", "MyNotificationChannelName", NotificationManager.IMPORTANCE_DEFAULT);
//                    notificationManager.createNotificationChannel(channel);
//                }
//                builder = new NotificationCompat.Builder(getApplicationContext(), "MyNotificationChannelName");
//            }else {
//                builder = new NotificationCompat.Builder(getApplicationContext());
//            }
//
//            String title = remoteMessage.getNotification().getTitle();
//            String body = remoteMessage.getNotification().getBody();
//
//            builder.setContentTitle(title)
//                    .setContentText(body)
//                    .setSmallIcon(R.drawable.ic_launcher_background);
//
//            Notification notification = builder.build();
//            notificationManager.notify(1, notification);
//        }

        @Override
        public void onMessageReceived(RemoteMessage remoteMessage) {
            super.onMessageReceived(remoteMessage);

            NotificationManagerCompat notificationManager = NotificationManagerCompat.from(getApplicationContext());


            // Check if message contains a data payload.
//            if (remoteMessage.getData().size() > 0) {
//                Map<String, String> data = remoteMessage.getData();
//                // You can access your data as follows, assuming "key1" and "key2" are keys in your data
//                String value1 = data.get("key1");
//                String value2 = data.get("key2");
//                // Handle the data message here.
//                Log.d(TAG, "Message data payload: " + remoteMessage.getData());
//            }

            // Check if message contains a notification payload.
            if (remoteMessage.getNotification() != null) {
                String title = remoteMessage.getNotification().getTitle();
                String body = remoteMessage.getNotification().getBody();

                // Create an explicit intent for an Activity in your app
                Intent intent = new Intent(this, MainActivity.class);
                intent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);

                // Add FLAG_IMMUTABLE
                int flags = PendingIntent.FLAG_UPDATE_CURRENT | (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M ? PendingIntent.FLAG_IMMUTABLE : 0);

                PendingIntent pendingIntent = PendingIntent.getActivity(this, 0, intent, flags);


                String channelId = "Default";
                NotificationCompat.Builder builder = new NotificationCompat.Builder(this, channelId)
                        .setSmallIcon(R.mipmap.ic_launcher)
                        .setContentTitle(title)
                        .setContentText(body)
                        .setAutoCancel(true)
                        .setContentIntent(pendingIntent);



                // For Android Oreo and later, a Notification Channel is needed.
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    // Create the NotificationChannel
                    String channelName = "Default Channel";
                    NotificationChannel channel = new NotificationChannel(channelId, channelName, NotificationManager.IMPORTANCE_DEFAULT);
//                    NotificationManager notificationManager = getSystemService(NotificationManager.class);
                    notificationManager.createNotificationChannel(channel);
                }

                notificationManager.notify(0, builder.build());
            }
        }
    }