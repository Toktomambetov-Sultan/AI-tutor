import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import api from '../../api/axios';
import './Admin.css';

export default function UserDetail() {
    const { id } = useParams();
    const navigate = useNavigate();
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => { fetchUser(); }, [id]);

    const fetchUser = async () => {
        try {
            const response = await api.get(`/admin/users/${id}`);
            setUser(response.data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to fetch user');
        } finally {
            setLoading(false);
        }
    };

    const handleDelete = async () => {
        if (!confirm('Are you sure you want to delete this user?')) return;
        try {
            await api.delete(`/admin/users/${id}`);
            navigate('/admin/users');
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to delete user');
        }
    };

    const handleRestore = async () => {
        try {
            await api.post(`/admin/users/${id}/restore`);
            fetchUser();
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to restore user');
        }
    };

    if (loading) return <div className="loading-state"><div className="spinner" /><span>Loading user...</span></div>;
    if (error && !user) return <div className="alert alert-error">{error}</div>;
    if (!user) return <div className="alert alert-error">User not found</div>;

    return (
        <div>
            <button className="btn-back" onClick={() => navigate('/admin/users')}>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M10 12L6 8l4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
                Back to Users
            </button>

            <div className="page-header">
                <h1>User Detail</h1>
            </div>

            <div className="card">
                {user.deleted_at && (
                    <div className="deleted-banner">
                        <strong>⚠ This user has been deleted</strong>
                        <span className="deleted-date">on {new Date(user.deleted_at).toLocaleString()}</span>
                    </div>
                )}

                <div className="detail-grid">
                    <div className="detail-item">
                        <label>Full Name</label>
                        <p>{user.full_name}</p>
                    </div>
                    <div className="detail-item">
                        <label>Email</label>
                        <p>{user.email}</p>
                    </div>
                    <div className="detail-item">
                        <label>Role</label>
                        <p><span className={`badge badge-${user.role}`}>{user.role}</span></p>
                    </div>
                    <div className="detail-item">
                        <label>Status</label>
                        <p>
                            {user.is_active
                                ? <span className="badge badge-active">Active</span>
                                : <span className="badge badge-inactive">Inactive</span>
                            }
                        </p>
                    </div>
                    <div className="detail-item">
                        <label>Created</label>
                        <p>{new Date(user.created_at).toLocaleString()}</p>
                    </div>
                    <div className="detail-item">
                        <label>Last Updated</label>
                        <p>{user.updated_at ? new Date(user.updated_at).toLocaleString() : '—'}</p>
                    </div>
                </div>

                <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid var(--border)' }}>
                    {user.deleted_at ? (
                        <button onClick={handleRestore} className="btn-primary">Restore User</button>
                    ) : (
                        <button onClick={handleDelete} className="btn-danger">Delete User</button>
                    )}
                </div>
            </div>
        </div>
    );
}
