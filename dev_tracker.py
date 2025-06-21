"""
Development Tracking Utility
Handles version tracking, changelog generation, and auto-commits for file changes.
"""

import os
import subprocess
import json
from datetime import datetime
from typing import List, Dict, Optional
import uuid
from sound_notifier import play_completion_notification


class DevTracker:
    def __init__(self, change_logs_dir: str = "change_logs"):
        self.change_logs_dir = change_logs_dir
        self.session_id = str(uuid.uuid4())[:8]
        
        # Ensure change_logs directory exists
        os.makedirs(change_logs_dir, exist_ok=True)
    
    def generate_changelog_filename(self, description: str) -> str:
        """Generate changelog filename with timestamp and description"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Clean description for filename
        clean_desc = "".join(c for c in description if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_desc = clean_desc.replace(' ', '_').lower()[:30]  # Limit length
        return f"{timestamp}_{clean_desc}.json"
    
    def get_git_status(self) -> Dict:
        """Get current git status"""
        try:
            # Get modified files
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, check=True)
            modified_files = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    status = line[:2]
                    filename = line[3:]
                    modified_files.append({"status": status.strip(), "file": filename})
            
            # Get current branch
            branch_result = subprocess.run(['git', 'branch', '--show-current'], 
                                         capture_output=True, text=True, check=True)
            current_branch = branch_result.stdout.strip()
            
            return {
                "modified_files": modified_files,
                "current_branch": current_branch
            }
        except subprocess.CalledProcessError as e:
            return {"error": f"Git command failed: {e}"}
    
    def create_commit(self, message: str) -> Dict:
        """Create a git commit with the given message"""
        try:
            # Add all modified files
            subprocess.run(['git', 'add', '.'], check=True)
            
            # Create commit
            commit_message = f"{message}\n\n🤖 Generated with Claude Code\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            
            # Get commit hash
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, check=True)
            commit_hash = result.stdout.strip()
            
            return {
                "success": True,
                "commit_hash": commit_hash,
                "message": message
            }
        except subprocess.CalledProcessError as e:
            return {"success": False, "error": f"Commit failed: {e}"}
    
    def save_changelog(self, user_prompt: str, description: str, 
                      files_changed: List[str], commit_info: Dict) -> str:
        """Save changelog entry to file"""
        timestamp = datetime.now().isoformat()
        filename = self.generate_changelog_filename(description)
        filepath = os.path.join(self.change_logs_dir, filename)
        
        changelog_entry = {
            "session_id": self.session_id,
            "timestamp": timestamp,
            "commit_time": timestamp,
            "user_prompt": user_prompt,
            "description": description,
            "files_changed": files_changed,
            "commit_info": commit_info,
            "git_status_before": self.get_git_status()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(changelog_entry, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def track_change(self, user_prompt: str, description: str, 
                    files_changed: List[str] = None, play_sound: bool = True) -> Dict:
        """Complete workflow: commit changes and create changelog"""
        # Get current git status
        git_status = self.get_git_status()
        
        # If no files specified, extract from git status
        if files_changed is None:
            files_changed = [item["file"] for item in git_status.get("modified_files", [])]
        
        # Create commit
        commit_info = self.create_commit(description)
        
        # Save changelog
        changelog_path = self.save_changelog(user_prompt, description, 
                                           files_changed, commit_info)
        
        # Play completion sound if requested
        if play_sound:
            try:
                play_completion_notification()
            except Exception as e:
                print(f"Sound notification failed: {e}")
        
        return {
            "changelog_path": changelog_path,
            "commit_info": commit_info,
            "files_changed": files_changed,
            "session_id": self.session_id
        }


# Global instance for easy access
tracker = DevTracker()


def track_development_change(user_prompt: str, description: str, 
                           files_changed: List[str] = None, play_sound: bool = True) -> Dict:
    """Convenience function to track a development change"""
    return tracker.track_change(user_prompt, description, files_changed, play_sound)


if __name__ == "__main__":
    # Example usage
    result = track_development_change(
        user_prompt="Add development tracking functionality",
        description="implement dev tracker utility",
        files_changed=["dev_tracker.py"]
    )
    print(f"Tracking result: {result}")