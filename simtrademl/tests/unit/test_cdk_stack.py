"""Unit tests for AWS CDK Stack CloudWatch configuration

Tests verify that:
- CloudWatch Log Groups are properly configured
- CloudWatch Alarms are created with correct thresholds
- SNS Topics are set up for alerting
- Alarm actions are properly connected
"""

import pytest
from aws_cdk import App, Stack
from aws_cdk import assertions as assert_cdk
from infra.cdk_stack import SimTradeMLStack


@pytest.fixture
def app():
    """Create CDK app for testing"""
    return App()


@pytest.fixture
def dev_stack(app):
    """Create development stack for testing"""
    return SimTradeMLStack(
        app,
        "TestStack",
        environment="dev",
        alert_email="test@example.com",
    )


@pytest.fixture
def prod_stack(app):
    """Create production stack for testing"""
    return SimTradeMLStack(
        app,
        "TestStackProd",
        environment="production",
        alert_email="ops@example.com",
    )


class TestCloudWatchLogGroup:
    """Test CloudWatch Log Group configuration"""

    def test_log_group_created(self, dev_stack):
        """Test that log group is created with correct name"""
        template = assert_cdk.Template.from_stack(dev_stack)

        template.has_resource_properties(
            "AWS::Logs::LogGroup",
            {
                "LogGroupName": "/ecs/simtrademl/dev",
                "RetentionInDays": 7,  # ONE_WEEK for dev
            },
        )

    def test_log_group_retention_by_environment(self, prod_stack):
        """Test that production has longer retention period"""
        template = assert_cdk.Template.from_stack(prod_stack)

        template.has_resource_properties(
            "AWS::Logs::LogGroup",
            {
                "LogGroupName": "/ecs/simtrademl/production",
                "RetentionInDays": 30,  # ONE_MONTH for production
            },
        )

    def test_log_group_count(self, dev_stack):
        """Test that exactly one log group is created"""
        template = assert_cdk.Template.from_stack(dev_stack)
        template.resource_count_is("AWS::Logs::LogGroup", 1)


class TestSNSTopic:
    """Test SNS Topic configuration for alerts"""

    def test_sns_topic_created(self, dev_stack):
        """Test that SNS topic is created"""
        template = assert_cdk.Template.from_stack(dev_stack)

        template.has_resource_properties(
            "AWS::SNS::Topic",
            {
                "TopicName": "simtrademl-alerts-dev",
                "DisplayName": "SimTradeML DEV Alerts",
            },
        )

    def test_email_subscription_created(self, dev_stack):
        """Test that email subscription is created when email provided"""
        template = assert_cdk.Template.from_stack(dev_stack)

        template.has_resource_properties(
            "AWS::SNS::Subscription",
            {
                "Protocol": "email",
                "Endpoint": "test@example.com",
            },
        )

    def test_sns_topic_count(self, dev_stack):
        """Test that exactly one SNS topic is created"""
        template = assert_cdk.Template.from_stack(dev_stack)
        template.resource_count_is("AWS::SNS::Topic", 1)


class TestCloudWatchAlarms:
    """Test CloudWatch Alarm configuration"""

    def test_high_error_rate_alarm_created(self, dev_stack):
        """Test that high error rate alarm is created"""
        template = assert_cdk.Template.from_stack(dev_stack)

        template.has_resource_properties(
            "AWS::CloudWatch::Alarm",
            {
                "AlarmName": "simtrademl-dev-high-error-rate",
                "AlarmDescription": "Alert when 5xx error rate exceeds 5%",
                "Threshold": 5.0,
                "EvaluationPeriods": 2,
                "DatapointsToAlarm": 2,
                "ComparisonOperator": "GreaterThanThreshold",
                "TreatMissingData": "notBreaching",
            },
        )

    def test_high_latency_alarm_created(self, dev_stack):
        """Test that high latency alarm is created"""
        template = assert_cdk.Template.from_stack(dev_stack)

        template.has_resource_properties(
            "AWS::CloudWatch::Alarm",
            {
                "AlarmName": "simtrademl-dev-high-latency",
                "AlarmDescription": "Alert when P95 latency exceeds 1000ms",
                "Threshold": 1.0,  # 1 second
                "EvaluationPeriods": 2,
                "DatapointsToAlarm": 2,
                "ComparisonOperator": "GreaterThanThreshold",
            },
        )

    def test_unhealthy_target_alarm_created(self, dev_stack):
        """Test that unhealthy target alarm is created"""
        template = assert_cdk.Template.from_stack(dev_stack)

        template.has_resource_properties(
            "AWS::CloudWatch::Alarm",
            {
                "AlarmName": "simtrademl-dev-unhealthy-targets",
                "AlarmDescription": "Alert when there are unhealthy targets",
                "Threshold": 1,
                "EvaluationPeriods": 3,
                "DatapointsToAlarm": 2,
                "ComparisonOperator": "GreaterThanOrEqualToThreshold",
            },
        )

    def test_high_cpu_alarm_created(self, dev_stack):
        """Test that high CPU utilization alarm is created"""
        template = assert_cdk.Template.from_stack(dev_stack)

        template.has_resource_properties(
            "AWS::CloudWatch::Alarm",
            {
                "AlarmName": "simtrademl-dev-high-cpu",
                "AlarmDescription": "Alert when CPU utilization exceeds 80%",
                "Threshold": 80,
                "EvaluationPeriods": 2,
                "DatapointsToAlarm": 2,
                "ComparisonOperator": "GreaterThanThreshold",
            },
        )

    def test_high_memory_alarm_created(self, dev_stack):
        """Test that high memory utilization alarm is created"""
        template = assert_cdk.Template.from_stack(dev_stack)

        template.has_resource_properties(
            "AWS::CloudWatch::Alarm",
            {
                "AlarmName": "simtrademl-dev-high-memory",
                "AlarmDescription": "Alert when memory utilization exceeds 80%",
                "Threshold": 80,
                "EvaluationPeriods": 2,
                "DatapointsToAlarm": 2,
                "ComparisonOperator": "GreaterThanThreshold",
            },
        )

    def test_low_request_count_alarm_created(self, dev_stack):
        """Test that low request count alarm is created"""
        template = assert_cdk.Template.from_stack(dev_stack)

        template.has_resource_properties(
            "AWS::CloudWatch::Alarm",
            {
                "AlarmName": "simtrademl-dev-low-requests",
                "AlarmDescription": "Alert when request count drops significantly",
                "Threshold": 10,
                "EvaluationPeriods": 3,
                "DatapointsToAlarm": 3,
                "ComparisonOperator": "LessThanThreshold",
                "TreatMissingData": "breaching",
            },
        )

    def test_composite_alarm_created(self, dev_stack):
        """Test that composite alarm is created for critical issues"""
        template = assert_cdk.Template.from_stack(dev_stack)

        template.has_resource_properties(
            "AWS::CloudWatch::CompositeAlarm",
            {
                "AlarmName": "simtrademl-dev-critical",
                "AlarmDescription": "Critical: Multiple issues detected",
            },
        )

    def test_alarm_count(self, dev_stack):
        """Test that all expected alarms are created"""
        template = assert_cdk.Template.from_stack(dev_stack)

        # 6 regular alarms + 1 composite alarm
        template.resource_count_is("AWS::CloudWatch::Alarm", 6)
        template.resource_count_is("AWS::CloudWatch::CompositeAlarm", 1)

    def test_alarms_have_sns_actions(self, dev_stack):
        """Test that alarms are connected to SNS topic"""
        template = assert_cdk.Template.from_stack(dev_stack)

        # Verify that alarms have AlarmActions pointing to SNS topic
        template.has_resource_properties(
            "AWS::CloudWatch::Alarm",
            {
                "AlarmActions": assert_cdk.Match.array_with(
                    [assert_cdk.Match.object_like({"Ref": assert_cdk.Match.string_like_regexp(".*AlertTopic.*")})]
                ),
            },
        )


class TestECSTaskLogging:
    """Test ECS Task Definition logging configuration"""

    def test_task_has_cloudwatch_logging(self, dev_stack):
        """Test that ECS task is configured with CloudWatch logging"""
        template = assert_cdk.Template.from_stack(dev_stack)

        template.has_resource_properties(
            "AWS::ECS::TaskDefinition",
            {
                "ContainerDefinitions": assert_cdk.Match.array_with(
                    [
                        assert_cdk.Match.object_like(
                            {
                                "LogConfiguration": {
                                    "LogDriver": "awslogs",
                                    "Options": {
                                        "awslogs-stream-prefix": "inference",
                                    },
                                }
                            }
                        )
                    ]
                ),
            },
        )


class TestEnvironmentSpecificConfiguration:
    """Test environment-specific configurations"""

    def test_dev_has_debug_logging(self, dev_stack):
        """Test that dev environment has DEBUG log level"""
        template = assert_cdk.Template.from_stack(dev_stack)

        template.has_resource_properties(
            "AWS::ECS::TaskDefinition",
            {
                "ContainerDefinitions": assert_cdk.Match.array_with(
                    [
                        assert_cdk.Match.object_like(
                            {
                                "Environment": assert_cdk.Match.array_with(
                                    [{"Name": "LOG_LEVEL", "Value": "DEBUG"}]
                                ),
                            }
                        )
                    ]
                ),
            },
        )

    def test_prod_has_info_logging(self, prod_stack):
        """Test that production environment has INFO log level"""
        template = assert_cdk.Template.from_stack(prod_stack)

        template.has_resource_properties(
            "AWS::ECS::TaskDefinition",
            {
                "ContainerDefinitions": assert_cdk.Match.array_with(
                    [
                        assert_cdk.Match.object_like(
                            {
                                "Environment": assert_cdk.Match.array_with(
                                    [{"Name": "LOG_LEVEL", "Value": "INFO"}]
                                ),
                            }
                        )
                    ]
                ),
            },
        )

    def test_prod_has_more_tasks(self, prod_stack):
        """Test that production has more desired tasks"""
        template = assert_cdk.Template.from_stack(prod_stack)

        template.has_resource_properties(
            "AWS::ECS::Service",
            {
                "DesiredCount": 2,  # Production should have at least 2
            },
        )

    def test_dev_has_one_task(self, dev_stack):
        """Test that dev environment has only 1 task"""
        template = assert_cdk.Template.from_stack(dev_stack)

        template.has_resource_properties(
            "AWS::ECS::Service",
            {
                "DesiredCount": 1,  # Dev should have 1
            },
        )


class TestStackTags:
    """Test that proper tags are applied"""

    def test_stack_has_required_tags(self, dev_stack):
        """Test that stack has all required tags"""
        tags = dev_stack.tags.tag_values()

        assert tags.get("Project") == "SimTradeML"
        assert tags.get("Environment") == "dev"
        assert tags.get("ManagedBy") == "CDK"
