"""
Log viewer utility for the Local LLM application.
Provides tools to view and analyze API logs.
"""
import os
import json
import argparse
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

class LogViewer:
    """Utility class for viewing and analyzing API logs."""
    
    def __init__(self, log_dir: str = './logs'):
        self.log_dir = log_dir
        self.api_log_file = os.path.join(log_dir, 'api.log')
        self.error_log_file = os.path.join(log_dir, 'errors.log')
    
    def parse_log_line(self, line: str) -> Optional[Dict]:
        """Parse a log line and return structured data."""
        try:
            return json.loads(line.strip())
        except json.JSONDecodeError:
            return None
    
    def read_logs(self, log_file: str, lines: int = None) -> List[Dict]:
        """Read and parse log entries."""
        if not os.path.exists(log_file):
            return []
        
        logs = []
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Split by JSON object boundaries (lines starting with '{')
            json_objects = []
            current_json = ""
            
            for line in content.split('\n'):
                if line.strip().startswith('{') and current_json:
                    # Start of new JSON object, parse the previous one
                    try:
                        parsed = json.loads(current_json)
                        json_objects.append(parsed)
                    except json.JSONDecodeError:
                        pass
                    current_json = line
                else:
                    current_json += line + '\n'
            
            # Don't forget the last JSON object
            if current_json.strip():
                try:
                    parsed = json.loads(current_json)
                    json_objects.append(parsed)
                except json.JSONDecodeError:
                    pass
            
            # Return the requested number of lines (most recent)
            if lines and len(json_objects) > lines:
                logs = json_objects[-lines:]
            else:
                logs = json_objects
        
        return logs
    
    def get_api_logs(self, lines: int = 5000) -> List[Dict]:
        """Get recent API logs."""
        return self.read_logs(self.api_log_file, lines)
    
    def get_error_logs(self, lines: int = 5000) -> List[Dict]:
        """Get recent error logs."""
        return self.read_logs(self.error_log_file, lines)
    
    def clear_logs(self) -> Dict[str, bool]:
        """Clear all log files and return status."""
        results = {}
        
        # Clear API logs
        try:
            if os.path.exists(self.api_log_file):
                with open(self.api_log_file, 'w', encoding='utf-8') as f:
                    pass  # Just open in write mode to clear the file
                results['api_logs_cleared'] = True
            else:
                results['api_logs_cleared'] = True  # File doesn't exist, so considered cleared
        except Exception as e:
            results['api_logs_cleared'] = False
            results['api_logs_error'] = str(e)
        
        # Clear error logs
        try:
            if os.path.exists(self.error_log_file):
                with open(self.error_log_file, 'w', encoding='utf-8') as f:
                    pass  # Just open in write mode to clear the file
                results['error_logs_cleared'] = True
            else:
                results['error_logs_cleared'] = True  # File doesn't exist, so considered cleared
        except Exception as e:
            results['error_logs_cleared'] = False
            results['error_logs_error'] = str(e)
        
        return results

    def filter_logs_by_time(self, logs: List[Dict], hours: int = 24) -> List[Dict]:
        """Filter logs by time period."""
        # Create IST timezone (UTC+5:30)
        ist_timezone = timezone(timedelta(hours=5, minutes=30))
        cutoff_time = datetime.now(ist_timezone) - timedelta(hours=hours)
        filtered = []
        
        for log in logs:
            try:
                log_time = datetime.fromisoformat(log.get('timestamp', '').replace('Z', '+00:00'))
                if log_time >= cutoff_time:
                    filtered.append(log)
            except (ValueError, TypeError):
                continue
        
        return filtered
    
    def filter_logs_by_endpoint(self, logs: List[Dict], endpoint: str) -> List[Dict]:
        """Filter logs by endpoint."""
        return [log for log in logs if log.get('endpoint') == endpoint]
    
    def filter_logs_by_status(self, logs: List[Dict], status_code: int) -> List[Dict]:
        """Filter logs by HTTP status code."""
        return [log for log in logs if log.get('status_code') == status_code]
    
    def get_error_summary(self, hours: int = 24) -> Dict:
        """Get summary of errors in the last N hours."""
        error_logs = self.get_error_logs(lines=1000)
        recent_errors = self.filter_logs_by_time(error_logs, hours)
        
        summary = {
            'total_errors': len(recent_errors),
            'error_types': {},
            'endpoints_with_errors': {},
            'status_codes': {}
        }
        
        for error in recent_errors:
            # Count error types
            error_type = error.get('error_type', 'Unknown')
            summary['error_types'][error_type] = summary['error_types'].get(error_type, 0) + 1
            
            # Count endpoints with errors
            endpoint = error.get('endpoint', 'Unknown')
            summary['endpoints_with_errors'][endpoint] = summary['endpoints_with_errors'].get(endpoint, 0) + 1
            
            # Count status codes
            status_code = error.get('status_code', 'Unknown')
            summary['status_codes'][status_code] = summary['status_codes'].get(status_code, 0) + 1
        
        return summary
    
    def get_api_stats(self, hours: int = 24) -> Dict:
        """Get API usage statistics."""
        api_logs = self.get_api_logs(lines=2000)
        recent_logs = self.filter_logs_by_time(api_logs, hours)
        
        stats = {
            'total_requests': 0,
            'endpoints': {},
            'methods': {},
            'status_codes': {},
            'avg_response_time': 0,
            'response_times': []
        }
        
        for log in recent_logs:
            if log.get('type') == 'response':
                stats['total_requests'] += 1
                
                # Count endpoints
                endpoint = log.get('endpoint', 'Unknown')
                stats['endpoints'][endpoint] = stats['endpoints'].get(endpoint, 0) + 1
                
                # Count methods
                method = log.get('method', 'Unknown')
                stats['methods'][method] = stats['methods'].get(method, 0) + 1
                
                # Count status codes
                status_code = log.get('status_code', 'Unknown')
                stats['status_codes'][status_code] = stats['status_codes'].get(status_code, 0) + 1
                
                # Track response times
                duration = log.get('duration_ms', 0)
                if duration:
                    stats['response_times'].append(duration)
        
        # Calculate average response time
        if stats['response_times']:
            stats['avg_response_time'] = sum(stats['response_times']) / len(stats['response_times'])
        
        return stats
    
    def print_logs(self, logs: List[Dict], limit: int = 10):
        """Print logs in a readable format."""
        for log in logs[-limit:]:
            timestamp = log.get('timestamp', 'Unknown')
            level = log.get('level', 'INFO')
            message = log.get('message', '')
            
            print(f"[{timestamp}] {level}: {message}")
            
            # Print additional details for API logs
            if log.get('type') in ['request', 'response']:
                method = log.get('method', '')
                url = log.get('url', '')
                status_code = log.get('status_code', '')
                duration = log.get('duration_ms', '')
                
                if log.get('type') == 'request':
                    print(f"  → {method} {url}")
                else:
                    print(f"  ← {status_code} ({duration}ms)")
            
            print()
    
    def print_error_summary(self, hours: int = 24):
        """Print error summary."""
        summary = self.get_error_summary(hours)
        
        print(f"📊 Error Summary (Last {hours} hours)")
        print("=" * 50)
        print(f"Total Errors: {summary['total_errors']}")
        
        if summary['error_types']:
            print("\n🔥 Error Types:")
            for error_type, count in sorted(summary['error_types'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {error_type}: {count}")
        
        if summary['endpoints_with_errors']:
            print("\n🎯 Endpoints with Errors:")
            for endpoint, count in sorted(summary['endpoints_with_errors'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {endpoint}: {count}")
        
        if summary['status_codes']:
            print("\n📈 Status Codes:")
            for status_code, count in sorted(summary['status_codes'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {status_code}: {count}")
    
    def print_api_stats(self, hours: int = 24):
        """Print API usage statistics."""
        stats = self.get_api_stats(hours)
        
        print(f"📈 API Statistics (Last {hours} hours)")
        print("=" * 50)
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Average Response Time: {stats['avg_response_time']:.2f}ms")
        
        if stats['endpoints']:
            print("\n🎯 Most Used Endpoints:")
            for endpoint, count in sorted(stats['endpoints'].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {endpoint}: {count}")
        
        if stats['methods']:
            print("\n📋 HTTP Methods:")
            for method, count in sorted(stats['methods'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {method}: {count}")
        
        if stats['status_codes']:
            print("\n📊 Status Codes:")
            for status_code, count in sorted(stats['status_codes'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {status_code}: {count}")

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='View and analyze Local LLM API logs')
    parser.add_argument('--log-dir', default='./logs', help='Log directory path')
    parser.add_argument('--lines', type=int, default=50, help='Number of recent log lines to show')
    parser.add_argument('--hours', type=int, default=24, help='Time period in hours for analysis')
    parser.add_argument('--type', choices=['api', 'errors', 'both'], default='both', help='Type of logs to view')
    parser.add_argument('--stats', action='store_true', help='Show statistics instead of raw logs')
    parser.add_argument('--errors-only', action='store_true', help='Show only error summary')
    
    args = parser.parse_args()
    
    viewer = LogViewer(args.log_dir)
    
    if args.errors_only:
        viewer.print_error_summary(args.hours)
    elif args.stats:
        viewer.print_api_stats(args.hours)
        print()
        viewer.print_error_summary(args.hours)
    else:
        if args.type in ['api', 'both']:
            print("📋 Recent API Logs:")
            print("=" * 50)
            api_logs = viewer.get_api_logs(args.lines)
            viewer.print_logs(api_logs, args.lines)
        
        if args.type in ['errors', 'both']:
            print("\n🔥 Recent Error Logs:")
            print("=" * 50)
            error_logs = viewer.get_error_logs(args.lines)
            viewer.print_logs(error_logs, args.lines)

if __name__ == '__main__':
    main()
