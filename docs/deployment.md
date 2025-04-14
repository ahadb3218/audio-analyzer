# Deployment Guide

This guide provides instructions for deploying the Audio Analyzer application in a production environment.

## Production Requirements

### Hardware Requirements
- CPU: 4+ cores
- RAM: 16GB minimum
- Storage: 100GB+ SSD
- GPU: NVIDIA GPU with 8GB+ VRAM (recommended)

### Software Requirements
- Python 3.9+
- CUDA 11.0+ (for GPU support)
- Docker (optional)
- Nginx
- Gunicorn

## Deployment Options

### 1. Docker Deployment

#### Dockerfile
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:create_app()"]
```

#### Docker Compose
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./uploads:/app/uploads
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
      - PORT=5000
      - HOST=0.0.0.0
      - DEBUG=False
      - WHISPER_MODEL=base
      - DEVICE=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - app
```

### 2. Traditional Deployment

#### System Setup
1. **Update System**
   ```bash
   sudo apt update && sudo apt upgrade
   ```

2. **Install Dependencies**
   ```bash
   sudo apt install python3.9 python3.9-venv nginx ffmpeg
   ```

3. **Create Application User**
   ```bash
   sudo useradd -m -s /bin/bash audioanalyzer
   ```

#### Application Setup
1. **Clone Repository**
   ```bash
   sudo -u audioanalyzer git clone https://github.com/yourusername/audioAnalyzer.git
   cd audioAnalyzer
   ```

2. **Create Virtual Environment**
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install gunicorn
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with production settings
   ```

#### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /path/to/static;
    }

    location /uploads {
        alias /path/to/uploads;
    }
}
```

#### Systemd Service
```ini
[Unit]
Description=Audio Analyzer Gunicorn Service
After=network.target

[Service]
User=audioanalyzer
Group=audioanalyzer
WorkingDirectory=/home/audioanalyzer/audioAnalyzer
Environment="PATH=/home/audioanalyzer/audioAnalyzer/venv/bin"
ExecStart=/home/audioanalyzer/audioAnalyzer/venv/bin/gunicorn --workers 3 --bind 127.0.0.1:5000 app:create_app()

[Install]
WantedBy=multi-user.target
```

## Security Considerations

### 1. SSL/TLS
- Use Let's Encrypt for SSL certificates
- Configure HTTPS in Nginx
- Enable HSTS

### 2. Firewall
```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 3. File Permissions
```bash
sudo chown -R audioanalyzer:audioanalyzer /path/to/app
sudo chmod -R 755 /path/to/app
sudo chmod -R 777 /path/to/uploads
```

## Monitoring

### 1. Logging
- Configure log rotation
- Set up log aggregation
- Monitor error rates

### 2. Performance Monitoring
- CPU usage
- Memory usage
- GPU utilization
- Response times

### 3. Health Checks
- Configure monitoring endpoints
- Set up alerts
- Monitor disk space

## Backup Strategy

### 1. Application Data
```bash
# Backup uploads directory
tar -czf uploads_backup.tar.gz /path/to/uploads

# Backup logs
tar -czf logs_backup.tar.gz /path/to/logs
```

### 2. Database (if applicable)
```bash
# Backup database
pg_dump -U user database > backup.sql
```

## Scaling

### 1. Horizontal Scaling
- Use load balancer
- Configure multiple workers
- Set up session management

### 2. Vertical Scaling
- Increase resources
- Optimize model loading
- Implement caching

## Maintenance

### 1. Updates
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt

# Restart service
sudo systemctl restart audioanalyzer
```

### 2. Cleanup
```bash
# Clean old uploads
find /path/to/uploads -type f -mtime +7 -delete

# Clean logs
find /path/to/logs -type f -mtime +30 -delete
```

## Troubleshooting

### Common Issues
1. **GPU Issues**
   - Check CUDA installation
   - Verify GPU drivers
   - Monitor GPU memory

2. **Performance Issues**
   - Check worker count
   - Monitor system resources
   - Optimize model loading

3. **Storage Issues**
   - Monitor disk space
   - Clean up old files
   - Implement file rotation

## Disaster Recovery

### 1. Backup Restoration
```bash
# Restore uploads
tar -xzf uploads_backup.tar.gz -C /path/to/uploads

# Restore logs
tar -xzf logs_backup.tar.gz -C /path/to/logs
```

### 2. Service Recovery
```bash
# Restart services
sudo systemctl restart nginx
sudo systemctl restart audioanalyzer
``` 