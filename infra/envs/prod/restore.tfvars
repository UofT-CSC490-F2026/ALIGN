# NOT auto-loaded. Use only AFTER the approved prod destruction.
# Keep this file in future prod plans after restoring; removing the snapshot
# input can propose replacing the restored database.
db_snapshot_identifier = "a2-snapshot-before-disaster"
db_deletion_protection = true
# Reserved for a future approved deletion; this does not create a snapshot now.
db_final_snapshot_identifier = "align-prod-final-after-restore-20261022-01"
lake_force_destroy           = false
