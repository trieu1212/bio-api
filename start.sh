#!bin/bash

: ${MONGO_DB_NAME:=bio-db}
: ${MONGO_USER:=trieu}
: ${MONGO_PASSWORD:=trieu}

sleep 10

mongo <<EOF
use ${MONGO_DB_NAME}
db.createUser({
  user: "${MONGO_USER}",
  pwd: "${MONGO_PASSWORD}",
  roles: [{ role: "readWrite", db: "${MONGO_DB_NAME}" }]
})

db.users.insertMany([
    {
        _id: ObjectId('6731b2fa8fd0b7fb8d714470'),
        username: 'trieu',
        password: '\$2b\$12\$5YMaLGu6RY8eIT3HnGW4dOS0wBXyYzVcBDAhH0o0u3WHUe2y5sitm',
        email: 'trieu@gmail.com',
        role: 'user',
        label: null
    },
    {
        _id: ObjectId('6731b51c8fd0b7fb8d714471'),
        username: 'tram',
        password: '\$2b\$12\$2RixAkJSzmouw8hd/tTAuuPVC.GyDJGS/23rIGVqMaCwWuUJ/q766',
        email: 'tram@gmail.com',
        role: 'user',
        label: null
    },
    {
        _id: ObjectId('6731b6378fd0b7fb8d714472'),
        username: 'khang',
        password: '\$2b\$12\$TvPR4p7P1XgHn9l5KHvBLOIVHHawcXI4k3/Xddo2eBFKLRRNT0wEG',
        email: 'khang@gmail.com',
        role: 'user',
        label: null
    },
    {
        _id: ObjectId('6731b7388fd0b7fb8d714473'),
        username: 'minh',
        password: '\$2b\$12\$3e915PqzP0in/X57pvSJIe0rayO/lcfiHPV9z1KgfBU0cdm0BkTMy',
        email: 'minh@gmail.com',
        role: 'user',
        label: null
    }
])
EOF