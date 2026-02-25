import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import api from '../../api/axios';
import './Teacher.css';

export default function CourseList() {
    const [courses, setCourses] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [showForm, setShowForm] = useState(false);
    const [title, setTitle] = useState('');
    const [description, setDescription] = useState('');

    const fetchCourses = async () => {
        try {
            const response = await api.get('/courses');
            setCourses(response.data.courses);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to fetch courses');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { fetchCourses(); }, []);

    const handleCreate = async (e) => {
        e.preventDefault();
        try {
            await api.post('/courses', { title, description });
            setTitle('');
            setDescription('');
            setShowForm(false);
            fetchCourses();
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to create course');
        }
    };

    const handleDelete = async (courseId) => {
        if (!confirm('Delete this course and all its contents?')) return;
        try {
            await api.delete(`/courses/${courseId}`);
            fetchCourses();
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to delete course');
        }
    };

    return (
        <div>
            <div className="page-header">
                <h1>My Courses</h1>
                <button className="btn-primary" onClick={() => setShowForm(!showForm)}>
                    {showForm ? 'Cancel' : '+ New Course'}
                </button>
            </div>

            {showForm && (
                <div className="inline-form-card" style={{ maxWidth: '520px', marginBottom: '1.5rem' }}>
                    <form onSubmit={handleCreate}>
                        <div className="form-group">
                            <label>Course Title</label>
                            <input type="text" value={title} onChange={e => setTitle(e.target.value)} placeholder="e.g., Introduction to Computer Science" required />
                        </div>
                        <div className="form-group">
                            <label>Description</label>
                            <textarea value={description} onChange={e => setDescription(e.target.value)} rows={3} placeholder="Optional course description..." />
                        </div>
                        <button type="submit" className="btn-primary">Create Course</button>
                    </form>
                </div>
            )}

            {error && <div className="alert alert-error">{error}<button className="alert-close" onClick={() => setError('')}>×</button></div>}

            {loading ? (
                <div className="loading-state"><div className="spinner" /><span>Loading courses...</span></div>
            ) : courses.length === 0 ? (
                <div className="empty-state-card">
                    <div className="empty-icon">📚</div>
                    <h3>No courses yet</h3>
                    <p>Create your first course to get started!</p>
                </div>
            ) : (
                <div className="course-grid">
                    {courses.map(course => (
                        <div key={course.id} className="card course-card">
                            <h3><Link to={`/teacher/courses/${course.id}`}>{course.title}</Link></h3>
                            <p className="course-desc">{course.description || 'No description provided'}</p>
                            <div className="course-footer">
                                <span className="course-date">Created {new Date(course.created_at).toLocaleDateString()}</span>
                                <button onClick={() => handleDelete(course.id)} className="btn-ghost-danger btn-sm">Delete</button>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
