const functions = require('firebase-functions');
const admin = require('firebase-admin');
// admin.initializeApp();
// var serviceAccount = require("C:/Users/yongju kim/functions/test-f8051-firebase-adminsdk-19xji-389fe8e5e6.json");
// var serviceAccount = require("gs://test-f8051.appspot.com/test-f8051-firebase-adminsdk-19xji-389fe8e5e6.json");


// admin.initializeApp({
//   credential: admin.credential.cert(serviceAccount),
//   databaseURL: "https://test-f8051-default-rtdb.firebaseio.com/"
// });



const serviceAccount = functions.config().service_account.key;

admin.initializeApp({
    // credential: admin.credential.cert(JSON.parse(serviceAccount)),
    credential: admin.credential.cert(serviceAccount),
    databaseURL: "https://test-f8051-default-rtdb.firebaseio.com/"
});

const db = admin.database();
const wearable = db.ref('wearable tokens');
const smartphone = db.ref('phone tokens');

exports.sendNotification = functions.database.ref('/DATA')
    .onWrite(async (change, context) => {
        // const soundType = context.params.sound_type;
        // const soundDirection = context.params.sound_direction;
        // console.log(`New Direction: ${soundType}, New Type: ${soundDirection}`);
        console.log('sendNotification functions')
        const sound_direction = change.after.val().sound_direction;
        const sound_type = change.after.val().sound_type;

        // 이전 sound_direction, sound_type 값
        const prev_sound_direction = change.before.val().sound_direction;
        const prev_sound_type = change.before.val().sound_type;

        // console.log('Previous Direction: ${prev_sound_direction}, Previous Type: ${prev_sound_type}');
        // console.log('New Direction: ${sound_direction}, New Type: ${sound_type}');
        console.log('Previous Direction: ', prev_sound_direction + ' New Type: ', prev_sound_type);
        console.log('New Direction: ', sound_direction + ' New Type: ', sound_type);

        // sound_direction 이나 sound_type 값이 변경되었을 경우 알림을 보냅니다.
        if (prev_sound_direction !== sound_direction || prev_sound_type !== sound_type) {
            // FCM을 이용해 알림을 보내는 코드를 작성합니다.
            const payload = {
                notification: {
                    title: 'Sound Data Changed',
                    body: `${sound_direction}, ${sound_type}`,
                    // body: 'New Direction: ' + sound_direction + 'New Type: ' + sound_type
                    // 필요에 따라 추가 필드를 포함시킬 수 있습니다.
                }
            };

            // 이 토큰은 앱에서 얻은 해당 사용자의 토큰이어야 합니다. 이를 Realtime Database나 Firestore에 저장할 수 있습니다.
            // 이 예에서는 'userDeviceToken'을 사용했지만, 실제로는 DB에서 토큰을 읽어오는 로직이 필요합니다.
            // const userDeviceToken = "e61lDLP9QEGOBPtqFs72F6:APA91bHAWTh3seceYZ1vHJx4sQ7csUbmKME_v52tyKIuoZ90LddeQSUpcwglgFYyKmgyCOQ8fFOAZ4lCktNYpd2gRwouz-cGcHrNDzYqjv-hQNqT8z2bih5GejfnfqwwL1q8s44eqxg2"; 

        wearable.once('value', (snapshot) => {
            const tokens = snapshot.val();
            const userDeviceToken = tokens;
            console.log("wearable tokens:", userDeviceToken);
            admin.messaging().sendToDevice(userDeviceToken, payload)
                .then((response) => {
                    console.log('Successfully sent message:', response);
                })
                .catch((error) => {
                    console.log('Error sending message:', error);
                });
        });

        smartphone.once('value', (snapshot) => {
            const tokens = snapshot.val();
            const userDeviceToken = tokens;
            console.log("phone tokens:", userDeviceToken);
            admin.messaging().sendToDevice(userDeviceToken, payload)
                .then((response) => {
                    console.log('Successfully sent message:', response);
                })
                .catch((error) => {
                    console.log('Error sending message:', error);
                });
        });

    }
});