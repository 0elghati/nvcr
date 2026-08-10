# NVCR AWS release-asset staging bucket

This CDK app creates one private S3 bucket used only as temporary staging for
large release assets before they are copied into GitHub Releases. The bucket is
not the public distribution channel.

Default account and region:

```text
account: <aws-account-id>
region:  eu-west-1
stack:   NvcrReleaseAssetsStack
bucket:  nvcr-release-assets-<aws-account-id>-eu-west-1
```

## What it creates

- private S3 bucket with public access blocked;
- S3-managed server-side encryption;
- SSL-only bucket policy;
- bucket-owner-enforced object ownership;
- versioning;
- lifecycle cleanup for temporary objects;
- CloudFormation outputs for bucket name, region, and example prefix.

The stack uses `RemovalPolicy.RETAIN`, so deleting the stack does not delete
staged artifacts by surprise.

## First-time account bootstrap

Install CDK and the Python dependencies, then bootstrap the account/region once:

```bash
cd infra/aws-cdk
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
npm install -g aws-cdk

aws sts get-caller-identity
cdk bootstrap aws://<aws-account-id>/eu-west-1
```

If you use a named AWS profile, add `--profile <profile>` to the `aws` and `cdk`
commands.

## Deploy

```bash
cd infra/aws-cdk
. .venv/bin/activate
cdk synth
cdk deploy NvcrReleaseAssetsStack
```

Optional context overrides:

```bash
cdk deploy NvcrReleaseAssetsStack \
  -c releaseAccount=<aws-account-id> \
  -c releaseRegion=eu-west-1 \
  -c bucketName=nvcr-release-assets-<aws-account-id>-eu-west-1 \
  -c assetRetentionDays=14
```

## Use with NVCR engine staging

After packaging an engine bundle, upload it to S3 and generate the
`engine_assets.txt` row with a presigned URL:

```bash
./scripts/stage_engine_release_asset.sh \
  --version 0.16.0 \
  --engine-dir build/engines/dcvcrt-720p \
  --s3-uri s3://nvcr-release-assets-<aws-account-id>-eu-west-1/engine-assets \
  --aws-region eu-west-1 \
  --presign-expires 604800 \
  --asset-manifest dist/nvcr-engine-assets.txt
```

Then pass the generated file to the GitHub Actions upload workflow or paste its
contents into the workflow-dispatch form. GitHub Actions downloads with the
presigned HTTPS URL, validates the archive, and uploads the final assets to the
GitHub Release.
