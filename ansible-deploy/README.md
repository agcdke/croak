# RAG Chatbot — AWS Deployment with Ansible

## Architecture

```
                    Internet
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
    ┌─────────────┐         ┌─────────────┐
    │  UI Server  │         │  API Server │
    │  t3.medium  │────────▶│ c5.2xlarge  │
    │             │  HTTP   │             │
    │  Streamlit  │         │  FastAPI    │
    │  :8501      │         │  :8000      │
    │  Nginx :80  │         │  FastMCP    │
    └─────────────┘         │  :8001      │
                            │  Ollama     │
                            │  :11434     │
                            │  ChromaDB   │
                            └──────┬──────┘
                                   │ boto3
                            ┌──────▼──────┐
                            │  S3 Bucket  │
                            │  uploads/   │
                            │  chroma_    │
                            │  backup/    │
                            └─────────────┘
```

**Why split into two EC2 instances?**
- API server needs 16+ GB RAM for Ollama to run models comfortably
- UI server only runs Streamlit — a t3.medium ($0.046/hr) is enough
- Separating them means you can restart/update the UI without touching Ollama

---

## Step 0 — Prerequisites on your local machine

### Install Ansible + AWS dependencies

```bash
# macOS
brew install python@3.11 ansible

# Ubuntu/Debian
sudo apt install python3.11 python3-pip
pip3 install ansible

# Install AWS Ansible collections and boto3
ansible-galaxy collection install amazon.aws community.aws
pip3 install boto3 botocore
```

### Install AWS CLI

```bash
# macOS
brew install awscli

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip && sudo ./aws/install
```

---

## Step 1 — Create an AWS IAM user for Ansible

Ansible needs an IAM user with permissions to create EC2, S3, IAM resources.

```bash
# 1. Open AWS Console → IAM → Users → Create user
#    Name: ansible-deployer
#    Access type: Programmatic access (Access Key)

# 2. Attach these managed policies:
#    - AmazonEC2FullAccess
#    - AmazonS3FullAccess
#    - IAMFullAccess

# 3. Download the Access Key CSV when prompted — you only see it once

# 4. Configure AWS CLI with these credentials
aws configure
# AWS Access Key ID: AKIAIOSFODNN7EXAMPLE
# AWS Secret Access Key: wJalrXUtnFEMI...
# Default region: us-east-1
# Default output: json

# Verify it works
aws sts get-caller-identity
```

---

## Step 2 — Clone this repo and configure variables

```bash
git clone https://github.com/YOUR_USERNAME/pdf-turtle-chatbot.git
cd pdf-turtle-chatbot

# Copy the Ansible deploy folder into your project
cp -r ansible-deploy/ .
cd ansible-deploy/
```

### Edit group_vars/all.yml

Open `group_vars/all.yml` and change:

```yaml
# REQUIRED changes:
s3_bucket:   rag-chatbot-storage-YOURNAME   # must be globally unique
app_repo:    https://github.com/YOUR_USERNAME/pdf-turtle-chatbot.git
aws_region:  us-east-1                      # your preferred region
```

### (Optional) Store secrets with Ansible Vault

```bash
# Create encrypted vault file for AWS credentials
ansible-vault create group_vars/vault.yml

# Add these lines in the editor that opens:
vault_aws_access_key: AKIAIOSFODNN7EXAMPLE
vault_aws_secret_key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

---

## Step 3 — Provision AWS infrastructure

```bash
# Export credentials (or use vault — see above)
export AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id)
export AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key)

# Run the provision playbook — creates EC2, S3, IAM, security groups
ansible-playbook playbooks/provision.yml

# Expected output at the end:
# API server IP : 54.123.45.67
# UI  server IP : 52.90.12.34
# inventory/hosts.yml updated automatically
```

This creates:
- SSH key pair → saved to `~/.ssh/rag-chatbot-key.pem`
- Security groups for API and UI servers
- IAM role + instance profile (EC2 → S3 access, no hardcoded keys)
- S3 bucket with versioning + 30-day lifecycle on old backups
- 2 EC2 instances with Elastic IPs
- Updates `inventory/hosts.yml` automatically

**Wait ~2 minutes** for EC2 instances to fully boot before proceeding.

---

## Step 4 — Deploy the application

```bash
# Deploy to both servers in one command
ansible-playbook playbooks/deploy.yml

# Or deploy separately
ansible-playbook playbooks/deploy.yml --limit api   # API server only
ansible-playbook playbooks/deploy.yml --limit ui    # UI server only
```

What this does on the **API server**:
1. Installs Python 3.11, git, nginx, awscli
2. Clones your repo
3. Creates Python venv + installs requirements.txt
4. Installs and starts Ollama
5. Pulls `llama3.2` and `nomic-embed-text` models *(takes 5–15 min)*
6. Restores uploads/ and chroma_db/ from S3 (empty on first deploy)
7. Writes `.env` file
8. Installs and starts `rag-api` and `rag-mcp` systemd services
9. Configures Nginx reverse proxy
10. Sets up S3 backup cron jobs

What this does on the **UI server**:
1. Installs Python 3.11, git, nginx, awscli
2. Clones your repo
3. Creates Python venv + installs requirements.txt
4. Restores uploads/ from S3
5. Writes `.env` pointing Ollama at the API server IP
6. Installs and starts `rag-ui` systemd service
7. Configures Nginx for Streamlit WebSocket proxying

---

## Step 5 — Access the app

```bash
# Get the IPs
cat inventory/hosts.yml

# Open in browser
# Streamlit UI:  http://<UI_SERVER_IP>
# FastAPI docs:  http://<API_SERVER_IP>/api/docs
# FastMCP:       http://<API_SERVER_IP>/mcp/
```

---

## Daily operations

### Check service status

```bash
# SSH into API server
ssh -i ~/.ssh/rag-chatbot-key.pem ubuntu@<API_IP>

# Check all services
sudo systemctl status ollama rag-api rag-mcp

# View logs
tail -f /opt/rag-chatbot/logs/api.log
tail -f /opt/rag-chatbot/logs/ollama.log

# SSH into UI server
ssh -i ~/.ssh/rag-chatbot-key.pem ubuntu@<UI_IP>
sudo systemctl status rag-ui
tail -f /opt/rag-chatbot/logs/ui.log
```

### Redeploy after code changes

```bash
# Push your code changes to git, then:
ansible-playbook playbooks/deploy.yml
```

### Manual S3 backup

```bash
ssh -i ~/.ssh/rag-chatbot-key.pem ubuntu@<API_IP>
/usr/local/bin/rag-s3-backup.sh uploads
/usr/local/bin/rag-s3-backup.sh chroma
```

### Restart services

```bash
sudo systemctl restart rag-api rag-mcp   # API server
sudo systemctl restart rag-ui            # UI server
```

---

## EC2 instance type guide

| Instance | vCPU | RAM  | Cost/hr | Recommendation |
|----------|------|------|---------|----------------|
| t3.large | 2    | 8 GB | $0.083  | Too small for Ollama — avoid |
| c5.2xlarge | 8  | 16 GB | $0.34  | Minimum comfortable for llama3.2 |
| c5.4xlarge | 16 | 32 GB | $0.68  | Recommended — headroom for larger models |
| g4dn.xlarge | 4 + GPU | 16 GB | $0.526 | GPU inference — 5–10x faster but needs CUDA setup |

For `nomic-embed-text` + `llama3.2` on CPU, **c5.2xlarge is the minimum**.
For faster responses or larger models (llama3:70b), use **c5.4xlarge** or **g4dn.xlarge**.

---

## Cost estimate (us-east-1, 24/7)

| Resource | Monthly cost |
|----------|-------------|
| c5.2xlarge (API) | ~$245 |
| t3.medium (UI)   | ~$33  |
| 2 Elastic IPs    | ~$7   |
| S3 (10 GB)       | ~$0.23|
| **Total**        | **~$285/month** |

To save money: stop EC2 instances when not in use, or use Spot instances for the API server (add `spot_price: "0.20"` to ec2_instance in provision.yml).
