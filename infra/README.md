# AWS CDK Infrastructure Deployment

This directory contains AWS CDK infrastructure code for deploying SimTradeML to AWS.

## Architecture Components

### CloudWatch Monitoring
- **Log Groups**: Centralized logging for ECS tasks at `/ecs/simtrademl/{environment}`
- **Alarms**:
  - High error rate (>5% 5xx responses)
  - High latency (P95 >1000ms)
  - Unhealthy targets
  - High CPU utilization (>80%)
  - High memory utilization (>80%)
  - Low request count (potential service degradation)
  - Composite alarm for critical issues
- **SNS Topics**: Email notifications for all alarms

### ECS Fargate
- Container orchestration with auto-scaling
- Health checks via `/health` endpoint
- Circuit breaker for automatic rollback
- CPU and memory-based scaling policies

### Application Load Balancer
- HTTP traffic distribution
- Health check integration
- Target group management

## Prerequisites

1. Install AWS CDK:
```bash
npm install -g aws-cdk
```

2. Install Python dependencies:
```bash
poetry install
```

3. Configure AWS credentials:
```bash
aws configure
```

4. Bootstrap CDK (first time only):
```bash
cdk bootstrap aws://ACCOUNT-ID/REGION
```

## Deployment

### Deploy to Development
```bash
cdk deploy --context environment=dev --context alert_email=dev@example.com
```

### Deploy to Production
```bash
cdk deploy --context environment=production --context alert_email=ops@example.com
```

### Synthesize CloudFormation Template
```bash
cdk synth --context environment=dev
```

### View Differences
```bash
cdk diff --context environment=production
```

### Destroy Stack
```bash
cdk destroy --context environment=dev
```

## Environment Variables

- `ENVIRONMENT`: Deployment environment (dev/staging/production)
- `ALERT_EMAIL`: Email address for CloudWatch alarm notifications
- `CDK_DEFAULT_ACCOUNT`: AWS account ID
- `CDK_DEFAULT_REGION`: AWS region (default: us-east-1)

## Testing

Run CDK stack tests:
```bash
poetry run pytest simtrademl/tests/unit/test_cdk_stack.py -v
```

## Monitoring

### CloudWatch Dashboards
After deployment, view metrics in CloudWatch console:
- ECS Container Insights
- ALB metrics
- Custom application metrics

### Log Queries
Query structured logs using CloudWatch Logs Insights:
```
fields @timestamp, @message
| filter @message like /ERROR/
| sort @timestamp desc
| limit 20
```

### Alarm Management
- All alarms send notifications to the configured SNS topic
- Subscribe additional endpoints via AWS SNS console
- Alarm thresholds can be adjusted in `cdk_stack.py`

## Cost Optimization

### Development Environment
- 1 ECS task (1 vCPU, 2GB RAM)
- 1 NAT Gateway
- 7-day log retention
- Estimated cost: ~$50-70/month

### Production Environment
- 2-10 ECS tasks (auto-scaling)
- 1 NAT Gateway
- 30-day log retention
- Estimated cost: ~$150-300/month (baseline)

## Security

- VPC with private subnets for ECS tasks
- Public subnets for ALB only
- Security groups restrict traffic to necessary ports
- ECR image scanning enabled
- Secrets should be stored in AWS Secrets Manager (not yet implemented)

## Next Steps

1. Implement Secrets Manager integration for sensitive configuration
2. Add SSL/TLS certificate to ALB listener
3. Set up CloudFront CDN for global distribution
4. Implement blue-green deployment strategy
5. Add custom CloudWatch dashboard
