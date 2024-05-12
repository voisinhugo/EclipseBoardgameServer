Copy paste this file in ```/etc/systemd/system/eclipse.service```

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
ExecStart=cd /home/pi/Eclipse/EclipseBoardgameServer/server.py && gunicorn --workers 3 --bind unix:eclipse-boardgame-server.sock -m 007 wsgi:app

[Install]
WantedBy=multi-user.target
```

Then enter

```systemctl start eclipse.server```

```systemctl enable eclipse.server```
