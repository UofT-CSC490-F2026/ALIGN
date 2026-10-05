# Network module (PR 2)

Implementation follows in PR 2. This directory deliberately has no resources yet.

Inputs: project, environment, aws_region, vpc_cidr, availability_zones,
public_subnet_cidrs, private_subnet_cidrs, allowed_ssh_cidrs, db_port.

Outputs: vpc_id, public_subnet_ids, private_subnet_ids,
ec2_security_group_id, rds_security_group_id.

Subnet output order must match availability_zones. The module owns both security
groups. Compute and database modules consume the IDs rather than recreating them.
