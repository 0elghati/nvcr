from __future__ import annotations

import re

import aws_cdk as cdk
from aws_cdk import CfnOutput, RemovalPolicy
from aws_cdk import aws_s3 as s3
from constructs import Construct


class NvcrReleaseAssetsStack(cdk.Stack):
    """Temporary private S3 staging bucket for NVCR release assets.

    The bucket is intentionally private. Maintainers upload large artifacts with
    AWS CLI and generate short-lived presigned GET URLs for GitHub Actions. The
    final public distribution point remains GitHub Releases.
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        release_account = str(self.account)
        release_region = str(self.region)
        default_bucket_name = f"nvcr-release-assets-{release_account}-{release_region}"
        bucket_name = self.node.try_get_context("bucketName") or default_bucket_name
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket_name):
            raise ValueError(f"invalid S3 bucket name: {bucket_name}")

        asset_retention_days = int(self.node.try_get_context("assetRetentionDays") or 14)
        noncurrent_retention_days = int(self.node.try_get_context("noncurrentRetentionDays") or 7)
        abort_multipart_days = int(self.node.try_get_context("abortIncompleteMultipartUploadDays") or 1)
        if asset_retention_days < 1 or noncurrent_retention_days < 1 or abort_multipart_days < 1:
            raise ValueError("retention and multipart cleanup days must be positive")

        bucket = s3.Bucket(
            self,
            "ReleaseAssetsBucket",
            bucket_name=bucket_name,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="ExpireTemporaryReleaseAssets",
                    enabled=True,
                    expiration=cdk.Duration.days(asset_retention_days),
                    noncurrent_version_expiration=cdk.Duration.days(noncurrent_retention_days),
                    abort_incomplete_multipart_upload_after=cdk.Duration.days(abort_multipart_days),
                )
            ],
        )

        CfnOutput(self, "BucketName", value=bucket.bucket_name)
        CfnOutput(self, "BucketRegion", value=release_region)
        CfnOutput(self, "ExampleS3Uri", value=f"s3://{bucket.bucket_name}/releases/")
