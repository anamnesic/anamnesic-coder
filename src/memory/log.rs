use std::path::PathBuf;
use rusqlite::Connection;
use chrono::Local;

pub struct LongTermMemory {
    conn: Connection,
}

impl LongTermMemory {
    pub fn new(db_path: PathBuf) -> anyhow::Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(&db_path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                summary TEXT,
                context TEXT
            );
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                decision TEXT,
                reason TEXT
            );"
        )?;
        Ok(LongTermMemory { conn })
    }

    pub fn save_session(&self, summary: &str, context: &str) -> anyhow::Result<()> {
        self.conn.execute(
            "INSERT INTO sessions (timestamp, summary, context) VALUES (?1, ?2, ?3)",
            rusqlite::params![Local::now().to_rfc3339(), summary, context],
        )?;
        Ok(())
    }

    pub fn save_decision(&self, decision: &str, reason: &str) -> anyhow::Result<()> {
        self.conn.execute(
            "INSERT INTO decisions (timestamp, decision, reason) VALUES (?1, ?2, ?3)",
            rusqlite::params![Local::now().to_rfc3339(), decision, reason],
        )?;
        Ok(())
    }

    pub fn get_recent_sessions(&self, limit: usize) -> anyhow::Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT timestamp, summary FROM sessions ORDER BY id DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map(rusqlite::params![limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut result = Vec::new();
        for row in rows { result.push(row?); }
        Ok(result)
    }
}
