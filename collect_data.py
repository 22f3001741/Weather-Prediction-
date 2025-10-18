import time
import subprocess
from datetime import datetime
import sys

def run_continuous_collection(interval_minutes=30):
    """
    Continuously collect weather data at specified intervals
    
    Args:
        interval_minutes: Time between collections (default 30 minutes)
    """
    print("="*60)
    print("🌍 Weather Data Continuous Collection System")
    print("="*60)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏰ Collection interval: {interval_minutes} minutes")
    print(f"🛑 Press Ctrl+C to stop\n")
    print("="*60)
    
    collection_count = 0
    failed_count = 0
    
    try:
        while True:
            collection_count += 1
            print(f"\n🔄 Collection #{collection_count}")
            print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            try:
                # Run the weather collection script
                result = subprocess.run(
                    ["python", "weather.py"],
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minute timeout
                )
                
                if result.returncode == 0:
                    print(result.stdout)
                    print("✅ Collection successful!")
                else:
                    failed_count += 1
                    print(f"❌ Collection failed (exit code: {result.returncode})")
                    print(result.stderr)
                
            except subprocess.TimeoutExpired:
                failed_count += 1
                print("❌ Collection timed out after 2 minutes")
            
            except Exception as e:
                failed_count += 1
                print(f"❌ Error during collection: {e}")
            
            # Statistics
            print(f"\n📊 Statistics:")
            print(f"   Total collections: {collection_count}")
            print(f"   Successful: {collection_count - failed_count}")
            print(f"   Failed: {failed_count}")
            print(f"   Success rate: {((collection_count - failed_count) / collection_count * 100):.1f}%")
            
            # Wait for next collection
            wait_seconds = interval_minutes * 60
            next_time = datetime.fromtimestamp(time.time() + wait_seconds)
            print(f"\n⏳ Waiting {interval_minutes} minutes...")
            print(f"⏰ Next collection at: {next_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 60)
            
            time.sleep(wait_seconds)
    
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("🛑 Collection stopped by user")
        print("="*60)
        print(f"📊 Final Statistics:")
        print(f"   Total collections: {collection_count}")
        print(f"   Successful: {collection_count - failed_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Duration: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        sys.exit(0)

if __name__ == "__main__":
    # For testing: collect every 5 minutes
    # For production: collect every 30 minutes
    run_continuous_collection(interval_minutes=30)