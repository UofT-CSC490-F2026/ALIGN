import copy
import hashlib
import io
import unittest
from unittest.mock import Mock

import recovery_data as data
import recovery_sql


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.body = b'{"synthetic":true}'
        self.row = dict(key='demo/task.json', backup_version='backup-v1', source_version='source-v1', sha256=hashlib.sha256(self.body).hexdigest())
        self.manifest = dict(schema_version=1, account=data.ACCOUNT, region=data.REGION, lake=data.LAKE, backup=data.BACKUP, objects=[self.row])
        self.s3 = Mock()
        self.s3.get_object.side_effect = lambda **kw: {'Body': io.BytesIO(self.body)}
        self.s3.get_paginator.return_value.paginate.return_value = [{}]

    def test_dry_run_has_no_writes(self):
        self.assertEqual(data.restore(self.s3, self.manifest)['writes'], 0)
        self.s3.copy_object.assert_not_called()

    def test_apply_copies_exact_version(self):
        self.assertTrue(data.restore(self.s3, self.manifest, True)['all_hashes_match'])
        call = self.s3.copy_object.call_args.kwargs
        self.assertEqual(call['CopySource']['VersionId'], 'backup-v1')
        self.assertEqual(call['Bucket'], data.LAKE)
        self.assertEqual(call['ExpectedBucketOwner'], data.ACCOUNT)

    def test_bad_backup_rejected_before_writes(self):
        self.s3.get_object.side_effect = lambda **kw: {'Body': io.BytesIO(b'corrupt')}
        with self.assertRaises(ValueError):
            data.restore(self.s3, self.manifest, True)
        self.s3.copy_object.assert_not_called()

    def test_nonempty_destination_rejected(self):
        self.s3.get_paginator.return_value.paginate.return_value = [{'Versions': [{'Key': 'existing'}]}]
        with self.assertRaises(ValueError):
            data.restore(self.s3, self.manifest, True)
        self.s3.copy_object.assert_not_called()

    def test_wrong_bucket_rejected(self):
        self.manifest['lake'] = data.BACKUP
        with self.assertRaises(ValueError):
            data.restore(self.s3, self.manifest, True)
        self.s3.get_object.assert_not_called()

    def test_duplicate_keys_rejected(self):
        self.manifest['objects'].append(copy.deepcopy(self.row))
        with self.assertRaises(ValueError):
            data.validate_manifest(self.manifest)

    def test_unversioned_backup_rejected(self):
        self.row['backup_version'] = 'null'
        with self.assertRaises(ValueError):
            data.validate_manifest(self.manifest)

    def test_history_protection_fails_closed(self):
        self.s3.get_paginator.return_value.paginate.return_value = [{'Versions': [{'IsLatest': False}]}]
        with self.assertRaises(ValueError):
            data.protect(self.s3, 'snapshot')

    def test_delete_marker_protection_fails_closed(self):
        self.s3.get_paginator.return_value.paginate.return_value = [{'DeleteMarkers': [{'Key': 'deleted'}]}]
        with self.assertRaises(ValueError):
            data.protect(self.s3, 'snapshot')

    def test_protect_records_exact_backup_version(self):
        self.s3.get_paginator.return_value.paginate.return_value = [{'Versions': [{'IsLatest': True, 'Key': self.row['key'], 'VersionId': 'source-v1'}]}]
        def response(**kwargs):
            backup = kwargs['Bucket'] == data.BACKUP
            return {'Body': io.BytesIO(self.body), 'VersionId': 'backup-v1' if backup else 'source-v1', 'ReplicationStatus': 'REPLICA' if backup else 'COMPLETED'}
        self.s3.get_object.side_effect = response
        result = data.protect(self.s3, 'snapshot')
        self.assertEqual(result['objects'], [self.row])

    def test_restored_corruption_is_reported(self):
        self.s3.get_object.side_effect = [{'Body': io.BytesIO(self.body)}, {'Body': io.BytesIO(b'corrupt')}]
        with self.assertRaises(ValueError):
            data.restore(self.s3, self.manifest, True)

    def test_sql_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            recovery_sql.compare({'task_id': 1}, {'task_id': 2})

    def test_json_type_change_rejected(self):
        with self.assertRaises(ValueError):
            recovery_sql.compare({'synthetic': True}, {'synthetic': 1})

    def test_sql_identical_passes(self):
        recovery_sql.compare({'task_id': 1}, {'task_id': 1})


if __name__ == '__main__':
    unittest.main()
