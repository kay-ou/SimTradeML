#!/usr/bin/env python3
"""CDK Application entry point for SimTradeML infrastructure

Usage:
    # Synthesize CloudFormation template
    cdk synth

    # Deploy to AWS
    cdk deploy --context environment=dev --context alert_email=admin@example.com

    # Deploy to production
    cdk deploy --context environment=production --context alert_email=ops@example.com
"""

import os
from aws_cdk import App, Environment
from infra.cdk_stack import SimTradeMLStack


app = App()

# Get configuration from context or environment variables
environment = app.node.try_get_context("environment") or os.getenv("ENVIRONMENT", "dev")
alert_email = app.node.try_get_context("alert_email") or os.getenv("ALERT_EMAIL")

# Get AWS account and region from environment or use defaults
account = os.getenv("CDK_DEFAULT_ACCOUNT")
region = os.getenv("CDK_DEFAULT_REGION", "us-east-1")

# Create stack
SimTradeMLStack(
    app,
    f"SimTradeMLStack-{environment}",
    environment=environment,
    alert_email=alert_email,
    env=Environment(account=account, region=region),
    description=f"SimTradeML ML inference service infrastructure ({environment})",
    tags={
        "Project": "SimTradeML",
        "Environment": environment,
        "ManagedBy": "CDK",
    },
)

app.synth()
