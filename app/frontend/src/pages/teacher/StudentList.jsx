import { useState, useEffect } from 'react';
import api from '../../api/axios';
import './Teacher.css';

export default function StudentList() {
    const [students, setStudents] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => { fetchStudents(); }, []);

    const fetchStudents = async () => {
        try {
            const response = await api.get('/teacher/students');
            setStudents(response.data.users);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to fetch students');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <div className="page-header">
                <h1>My Students</h1>
            </div>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '1.25rem', fontSize: '0.875rem' }}>
                Students who have accessed your course materials.
            </p>

            {error && <div className="alert alert-error">{error}</div>}

            {loading ? (
                <div className="loading-state"><div className="spinner" /><span>Loading students...</span></div>
            ) : students.length === 0 ? (
                <div className="empty-state-card">
                    <div className="empty-icon">🎓</div>
                    <h3>No students yet</h3>
                    <p>Students will appear here once they access your lessons.</p>
                </div>
            ) : (
                <>
                    {/* Desktop */}
                    <div className="table-wrapper">
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Name</th>
                                    <th>Email</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {students.map(s => (
                                    <tr key={s.id}>
                                        <td style={{ fontWeight: 500 }}>{s.full_name}</td>
                                        <td style={{ color: 'var(--text-secondary)' }}>{s.email}</td>
                                        <td>
                                            {s.is_active
                                                ? <span className="badge badge-active">Active</span>
                                                : <span className="badge badge-inactive">Inactive</span>
                                            }
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    {/* Mobile */}
                    <div className="user-cards-mobile">
                        {students.map(s => (
                            <div key={s.id} className="user-card-mobile">
                                <div style={{ fontWeight: 600 }}>{s.full_name}</div>
                                <div style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>{s.email}</div>
                                <div style={{ marginTop: '0.5rem' }}>
                                    {s.is_active
                                        ? <span className="badge badge-active">Active</span>
                                        : <span className="badge badge-inactive">Inactive</span>
                                    }
                                </div>
                            </div>
                        ))}
                    </div>
                </>
            )}
        </div>
    );
}
