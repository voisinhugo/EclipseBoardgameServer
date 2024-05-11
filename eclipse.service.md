Copy paste this file in /etc/systemd/system/eclipse.service
Then enter
```systemctl start eclipse.server```
```systemctl enable eclipse.server```

[Unit]
Description=Flask server for eclipse webapp
After=network.target
StartLimitIntervalSec=0
[Service]
Type=simple
Restart=always
RestartSec=1
User=pi
ExecStart=/usr/bin/python3 /home/pi/Eclipse/EclipseBoardgameServer/server.py

[Install]
WantedBy=multi-user.target

