import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import api from '../../api/axios';
import './Admin.css';

export default function UserManagement() {
    const [users, setUsers] = useState([]);
    const [total, setTotal] = useState(0);
    const [roleFilter, setRoleFilter] = useState('');
    const [showDeleted, setShowDeleted] = useState(false);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    const fetchUsers = async () => {
        setLoading(true);
        try {
            const params = {};
            if (roleFilter) params.role = roleFilter;
            if (showDeleted) params.deleted = true;
            const response = await api.get('/admin/users', { params });
            setUsers(response.data.users);
            setTotal(response.data.total);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to fetch users');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchUsers();
    }, [roleFilter, showDeleted]);

    const handleDelete = async (userId) => {
        if (!confirm('Are you sure you want to delete this user?')) return;
        try {
            await api.delete(`/admin/users/${userId}`);
            fetchUsers();
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to delete user');
        }
    };

    const handleRestore = async (userId) => {
        try {
            await api.post(`/admin/users/${userId}/restore`);
            fetchUsers();
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to restore user');
        }
    };

    return (
        <div>
            <div className="page-header">
                <h1>User Management</h1>
                <Link to="/admin/users/new" className="btn-primary" style={{ textDecoration: 'none' }}>
                    + Create User
                </Link>
            </div>

            <div className="filters">
                <select value={roleFilter} onChange={e => setRoleFilter(e.target.value)}>
                    <option value="">All Roles</option>
                    <option value="admin">Admin</option>
                    <option value="teacher">Teacher</option>
                    <option value="student">Student</option>
                </select>
                <label className="checkbox-label">
                    <input type="checkbox" checked={showDeleted} onChange={e => setShowDeleted(e.target.checked)} />
                    Show deleted users
                </label>
            </div>

            {error && <div className="alert alert-error">{error}<button className="alert-close" onClick={() => setError('')}>×</button></div>}

            {loading ? (
                <div className="loading-state"><div className="spinner" /><span>Loading users...</span></div>
            ) : (
                <>
                    <p className="result-count">{total} user{total !== 1 ? 's' : ''} found</p>

                    {/* Desktop table */}
                    <div className="table-wrapper">
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Name</th>
                                    <th>Email</th>
                                    <th>Role</th>
                                    <th>Status</th>
                                    <th>Created</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {users.map(user => (
                                    <tr key={user.id} className={user.deleted_at ? 'row-deleted' : ''}>
                                        <td><Link to={`/admin/users/${user.id}`} style={{ fontWeight: 500 }}>{user.full_name}</Link></td>
                                        <td style={{ color: 'var(--text-secondary)' }}>{user.email}</td>
                                        <td><span className={`badge badge-${user.role}`}>{user.role}</span></td>
                                        <td>
                                            {user.deleted_at
                                                ? <span className="badge badge-deleted">Deleted</span>
                                                : user.is_active
                                                    ? <span className="badge badge-active">Active</span>
                                                    : <span className="badge badge-inactive">Inactive</span>
                                            }
                                        </td>
                                        <td style={{ color: 'var(--text-muted)' }}>{new Date(user.created_at).toLocaleDateString()}</td>
                                        <td>
                                            {user.deleted_at ? (
                                                <button onClick={() => handleRestore(user.id)} className="btn-ghost btn-sm">Restore</button>
                                            ) : (
                                                <button onClick={() => handleDelete(user.id)} className="btn-ghost-danger btn-sm">Delete</button>
                                            )}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    {/* Mobile cards */}
                    <div className="user-cards-mobile">
                        {users.map(user => (
                            <div key={user.id} className={`user-card-mobile ${user.deleted_at ? 'row-deleted' : ''}`}>
                                <div className="user-card-mobile-header">
                                    <div>
                                        <Link to={`/admin/users/${user.id}`} style={{ fontWeight: 600, fontSize: '0.9375rem' }}>{user.full_name}</Link>
                                        <div style={{ marginTop: '0.25rem' }}>
                                            <span className={`badge badge-${user.role}`}>{user.role}</span>{' '}
                                            {user.deleted_at
                                                ? <span className="badge badge-deleted">Deleted</span>
                                                : user.is_active
                                                    ? <span className="badge badge-active">Active</span>
                                                    : <span className="badge badge-inactive">Inactive</span>
                                            }
                                        </div>
                                    </div>
                                </div>
                                <div className="user-card-mobile-meta">
                                    <span>{user.email}</span>
                                    <span>Joined {new Date(user.created_at).toLocaleDateString()}</span>
                                </div>
                                <div className="user-card-mobile-actions">
                                    {user.deleted_at ? (
                                        <button onClick={() => handleRestore(user.id)} className="btn-secondary btn-sm">Restore</button>
                                    ) : (
                                        <button onClick={() => handleDelete(user.id)} className="btn-danger btn-sm">Delete</button>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </>
            )}
        </div>
    );
}
