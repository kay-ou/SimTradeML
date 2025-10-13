"""AWS CDK Stack for SimTradeML deployment with CloudWatch monitoring and alerting

This module defines the infrastructure for deploying SimTradeML as a containerized
service on AWS ECS Fargate with comprehensive monitoring and alerting.
"""

from typing import Optional

from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    aws_ecs as ecs,
    aws_ec2 as ec2,
    aws_elbv2 as elbv2,
    aws_ecr as ecr,
    aws_logs as logs,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cloudwatch_actions,
    aws_sns as sns,
    aws_sns_subscriptions as sns_subscriptions,
)
from constructs import Construct


class SimTradeMLStack(Stack):
    """Main CDK Stack for SimTradeML infrastructure

    This stack provisions:
    - VPC with public and private subnets
    - ECS Fargate cluster for running inference containers
    - Application Load Balancer for traffic distribution
    - ECR repository for Docker images
    - CloudWatch Log Groups for structured logging
    - CloudWatch Alarms for monitoring (error rate, latency, health)
    - SNS Topics for alert notifications
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        environment: str = "dev",
        alert_email: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the SimTradeML Stack

        Args:
            scope: CDK app scope
            construct_id: Unique identifier for this stack
            environment: Deployment environment (dev/staging/production)
            alert_email: Email address for receiving alerts
            **kwargs: Additional stack properties
        """
        super().__init__(scope, construct_id, **kwargs)

        self.environment = environment
        self.alert_email = alert_email

        # Create VPC
        self.vpc = self._create_vpc()

        # Create ECR Repository
        self.repository = self._create_ecr_repository()

        # Create CloudWatch Log Group
        self.log_group = self._create_log_group()

        # Create SNS Topic for alerts
        self.alert_topic = self._create_sns_topic()

        # Create ECS Cluster
        self.cluster = self._create_ecs_cluster()

        # Create Task Definition with logging
        self.task_definition = self._create_task_definition()

        # Create Fargate Service
        self.service = self._create_fargate_service()

        # Create Application Load Balancer
        self.load_balancer, self.target_group = self._create_load_balancer()

        # Create CloudWatch Alarms
        self._create_cloudwatch_alarms()

    def _create_vpc(self) -> ec2.Vpc:
        """Create VPC with public and private subnets"""
        return ec2.Vpc(
            self,
            "SimTradeMLVPC",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
            ],
        )

    def _create_ecr_repository(self) -> ecr.Repository:
        """Create ECR repository for Docker images"""
        return ecr.Repository(
            self,
            "SimTradeMLRepo",
            repository_name=f"simtrademl-{self.environment}",
            removal_policy=RemovalPolicy.RETAIN,
            image_scan_on_push=True,
        )

    def _create_log_group(self) -> logs.LogGroup:
        """Create CloudWatch Log Group for application logs

        The log group is configured with:
        - Structured log stream naming (by task ID)
        - Appropriate retention period based on environment
        - JSON log format for easy querying
        """
        retention_days = {
            "dev": logs.RetentionDays.ONE_WEEK,
            "staging": logs.RetentionDays.TWO_WEEKS,
            "production": logs.RetentionDays.ONE_MONTH,
        }

        return logs.LogGroup(
            self,
            "SimTradeMLLogGroup",
            log_group_name=f"/ecs/simtrademl/{self.environment}",
            retention=retention_days.get(
                self.environment, logs.RetentionDays.ONE_WEEK
            ),
            removal_policy=RemovalPolicy.DESTROY
            if self.environment == "dev"
            else RemovalPolicy.RETAIN,
        )

    def _create_sns_topic(self) -> sns.Topic:
        """Create SNS Topic for alert notifications

        Returns:
            SNS Topic that will be used for CloudWatch alarm notifications
        """
        topic = sns.Topic(
            self,
            "SimTradeMLAlertTopic",
            topic_name=f"simtrademl-alerts-{self.environment}",
            display_name=f"SimTradeML {self.environment.upper()} Alerts",
        )

        # Subscribe email if provided
        if self.alert_email:
            topic.add_subscription(
                sns_subscriptions.EmailSubscription(self.alert_email)
            )

        return topic

    def _create_ecs_cluster(self) -> ecs.Cluster:
        """Create ECS Cluster"""
        return ecs.Cluster(
            self,
            "SimTradeMLCluster",
            cluster_name=f"simtrademl-cluster-{self.environment}",
            vpc=self.vpc,
            container_insights=True,  # Enable CloudWatch Container Insights
        )

    def _create_task_definition(self) -> ecs.FargateTaskDefinition:
        """Create ECS Task Definition with logging configuration"""
        task_definition = ecs.FargateTaskDefinition(
            self,
            "InferenceTask",
            memory_limit_mib=2048,
            cpu=512,
        )

        # Add container with CloudWatch logging
        container = task_definition.add_container(
            "InferenceContainer",
            image=ecs.ContainerImage.from_ecr_repository(self.repository),
            logging=ecs.LogDriver.aws_logs(
                stream_prefix="inference",
                log_group=self.log_group,
            ),
            environment={
                "ENVIRONMENT": self.environment,
                "LOG_LEVEL": "INFO" if self.environment == "production" else "DEBUG",
            },
            # Add health check
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                retries=3,
                start_period=Duration.seconds(60),
            ),
        )

        container.add_port_mappings(ecs.PortMapping(container_port=8000))

        return task_definition

    def _create_fargate_service(self) -> ecs.FargateService:
        """Create Fargate Service with auto-scaling"""
        service = ecs.FargateService(
            self,
            "InferenceService",
            cluster=self.cluster,
            task_definition=self.task_definition,
            desired_count=2 if self.environment == "production" else 1,
            min_healthy_percent=50,
            max_healthy_percent=200,
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
        )

        # Configure auto-scaling
        scaling = service.auto_scale_task_count(
            min_capacity=1,
            max_capacity=10 if self.environment == "production" else 3,
        )

        # Scale based on CPU utilization
        scaling.scale_on_cpu_utilization(
            "CPUScaling",
            target_utilization_percent=70,
            scale_in_cooldown=Duration.seconds(60),
            scale_out_cooldown=Duration.seconds(60),
        )

        return service

    def _create_load_balancer(
        self,
    ) -> tuple[elbv2.ApplicationLoadBalancer, elbv2.ApplicationTargetGroup]:
        """Create Application Load Balancer with health checks"""
        # Create ALB
        alb = elbv2.ApplicationLoadBalancer(
            self,
            "LoadBalancer",
            vpc=self.vpc,
            internet_facing=True,
        )

        # Create target group
        target_group = elbv2.ApplicationTargetGroup(
            self,
            "InferenceTargetGroup",
            vpc=self.vpc,
            port=8000,
            protocol=elbv2.ApplicationProtocol.HTTP,
            targets=[self.service],
            health_check=elbv2.HealthCheck(
                path="/health",
                interval=Duration.seconds(30),
                timeout=Duration.seconds(5),
                healthy_threshold_count=2,
                unhealthy_threshold_count=3,
            ),
            deregistration_delay=Duration.seconds(30),
        )

        # Create listener (automatically attached to ALB)
        alb.add_listener(
            "Listener",
            port=80,
            default_target_groups=[target_group],
        )

        return alb, target_group

    def _create_cloudwatch_alarms(self) -> None:
        """Create CloudWatch alarms for monitoring system health

        Creates alarms for:
        1. High error rate (5xx responses > 5%)
        2. High latency (P95 > 1000ms)
        3. Low health score (unhealthy targets)
        4. High CPU utilization
        5. High memory utilization
        """
        alarm_action = cloudwatch_actions.SnsAction(self.alert_topic)

        # 1. High Error Rate Alarm
        error_rate_metric = cloudwatch.MathExpression(
            expression="(m1/m2) * 100",
            using_metrics={
                "m1": self.target_group.metric_http_code_target(
                    code=elbv2.HttpCodeTarget.TARGET_5XX_COUNT,
                    period=Duration.minutes(5),
                    statistic="Sum",
                ),
                "m2": self.target_group.metric_request_count(
                    period=Duration.minutes(5),
                    statistic="Sum",
                ),
            },
            label="Error Rate (%)",
        )

        error_rate_alarm = cloudwatch.Alarm(
            self,
            "HighErrorRateAlarm",
            alarm_name=f"simtrademl-{self.environment}-high-error-rate",
            alarm_description="Alert when 5xx error rate exceeds 5%",
            metric=error_rate_metric,
            threshold=5.0,
            evaluation_periods=2,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        error_rate_alarm.add_alarm_action(alarm_action)

        # 2. High Latency Alarm (Target Response Time)
        latency_alarm = cloudwatch.Alarm(
            self,
            "HighLatencyAlarm",
            alarm_name=f"simtrademl-{self.environment}-high-latency",
            alarm_description="Alert when P95 latency exceeds 1000ms",
            metric=self.target_group.metric_target_response_time(
                period=Duration.minutes(5),
                statistic="p95",
            ),
            threshold=1.0,  # 1 second
            evaluation_periods=2,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
        )
        latency_alarm.add_alarm_action(alarm_action)

        # 3. Unhealthy Target Alarm
        unhealthy_target_alarm = cloudwatch.Alarm(
            self,
            "UnhealthyTargetAlarm",
            alarm_name=f"simtrademl-{self.environment}-unhealthy-targets",
            alarm_description="Alert when there are unhealthy targets",
            metric=self.target_group.metric_unhealthy_host_count(
                period=Duration.minutes(1),
                statistic="Average",
            ),
            threshold=1,
            evaluation_periods=3,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
        )
        unhealthy_target_alarm.add_alarm_action(alarm_action)

        # 4. High CPU Utilization Alarm
        cpu_alarm = cloudwatch.Alarm(
            self,
            "HighCPUAlarm",
            alarm_name=f"simtrademl-{self.environment}-high-cpu",
            alarm_description="Alert when CPU utilization exceeds 80%",
            metric=self.service.metric_cpu_utilization(
                period=Duration.minutes(5),
                statistic="Average",
            ),
            threshold=80,
            evaluation_periods=2,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
        )
        cpu_alarm.add_alarm_action(alarm_action)

        # 5. High Memory Utilization Alarm
        memory_alarm = cloudwatch.Alarm(
            self,
            "HighMemoryAlarm",
            alarm_name=f"simtrademl-{self.environment}-high-memory",
            alarm_description="Alert when memory utilization exceeds 80%",
            metric=self.service.metric_memory_utilization(
                period=Duration.minutes(5),
                statistic="Average",
            ),
            threshold=80,
            evaluation_periods=2,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
        )
        memory_alarm.add_alarm_action(alarm_action)

        # 6. Low Request Count Alarm (potential service degradation)
        low_request_alarm = cloudwatch.Alarm(
            self,
            "LowRequestCountAlarm",
            alarm_name=f"simtrademl-{self.environment}-low-requests",
            alarm_description="Alert when request count drops significantly",
            metric=self.target_group.metric_request_count(
                period=Duration.minutes(5),
                statistic="Sum",
            ),
            threshold=10,
            evaluation_periods=3,
            datapoints_to_alarm=3,
            comparison_operator=cloudwatch.ComparisonOperator.LESS_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.BREACHING,
        )
        low_request_alarm.add_alarm_action(alarm_action)

        # Create composite alarm for critical issues
        composite_alarm = cloudwatch.CompositeAlarm(
            self,
            "CriticalIssueAlarm",
            composite_alarm_name=f"simtrademl-{self.environment}-critical",
            alarm_description="Critical: Multiple issues detected",
            alarm_rule=cloudwatch.AlarmRule.any_of(
                cloudwatch.AlarmRule.from_alarm(error_rate_alarm, cloudwatch.AlarmState.ALARM),
                cloudwatch.AlarmRule.from_alarm(unhealthy_target_alarm, cloudwatch.AlarmState.ALARM),
            ),
        )
        composite_alarm.add_alarm_action(alarm_action)
