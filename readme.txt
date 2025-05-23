================ Make nginx configs ==============
sudo nano /etc/nginx/sites-available/speech

    server {
        listen 80;
        return 301 https://$host$request_uri;
    }

    server {
        listen 443 ssl;
        server_name _; # wildcard for IP access
        client_max_body_size 100M;

        ssl_certificate /etc/ssl/certs/selfsigned.crt;
        ssl_certificate_key /etc/ssl/private/selfsigned.key;

        location /speech-processing/ {
            proxy_pass http://127.0.0.1:8000/;

            proxy_read_timeout 600;
            proxy_connect_timeout 600;
            proxy_send_timeout 600;

            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /test-sp-be/ {
            proxy_pass http://127.0.0.1:7500/;

            proxy_read_timeout 600;
            proxy_connect_timeout 600;
            proxy_send_timeout 600;

            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        access_log /var/log/nginx/flask.access.log;
        error_log /var/log/nginx/flask.error.log;
    }

sudo nginx -t && sudo systemctl reload nginx


================ Make Service ==================
sudo nano /etc/systemd/system/speech-processing-app.service

    [Unit]
    Description=Speech Processing App (FastAPI)
    After=network.target

    [Service]
    User=moshdev2213
    WorkingDirectory=/home/moshdev2213/test-sp-be
    ExecStart=/home/moshdev2213/test-sp-be/test-sp-env/bin/uvicorn app:app --host 0.0.0.0 --port 8000
    Restart=always
    RestartSec=3
    Environment=PYTHONUNBUFFERED=1

    [Install]
    WantedBy=multi-user.target

sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable speech-processing-app.service
sudo systemctl start speech-processing-app.service

sudo systemctl status speech-processing-app.service


========== clone repo ==========
git clone https://github.com/rp-org/test-sp-be.git
==========git pull lfs ===========
git pull lfs

===== dlib essesntials ======
sudo apt update
sudo apt install build-essential cmake libboost-all-dev python3-dev
