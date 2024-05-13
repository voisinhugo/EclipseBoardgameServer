# Eclipse Server Service

Copy paste this file in ```/etc/systemd/system/eclipseServer.service```

```service
[Unit]
Description=Gunicorn instance to server the Eclipse webapp
After=network.target
StartLimitIntervalSec=0
[Service]
Type=simple
Restart=always
RestartSec=1
User=pi
WorkingDirectory=/home/pi/Eclipse/EclipseBoardgameServer
ExecStart=/usr/bin/gunicorn --workers 3 --bind unix:eclipse-boardgame-server.sock --timeout 1000 wsgi:app

[Install]
WantedBy=multi-user.target
```

Then enter

```systemctl start eclipseServer.service```

```systemctl enable eclipseServer.service```
