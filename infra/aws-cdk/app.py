#!/usr/bin/env python3
import os

import aws_cdk as cdk

from nvcr_release_assets_stack import NvcrReleaseAssetsStack


app = cdk.App()
account = app.node.try_get_context("releaseAccount") or os.environ.get("CDK_DEFAULT_ACCOUNT")
region = app.node.try_get_context("releaseRegion") or os.environ.get("CDK_DEFAULT_REGION") or "eu-west-1"

if not account:
    raise ValueError("releaseAccount CDK context or CDK_DEFAULT_ACCOUNT is required")

NvcrReleaseAssetsStack(
    app,
    "NvcrReleaseAssetsStack",
    env=cdk.Environment(account=str(account), region=str(region)),
)

app.synth()
