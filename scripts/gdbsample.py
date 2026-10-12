# gdb -batch -x gdbsample.py : poor-man's sampling profiler. External loop sends SIGINT to the inferior.
import gdb
def on_stop(ev):
    if isinstance(ev, gdb.SignalEvent) and ev.stop_signal == 'SIGINT':
        for t in gdb.selected_inferior().threads():
            t.switch()
            if t.num == 1:
                fr = []; f = gdb.newest_frame()
                for i in range(14):
                    if f is None: break
                    fr.append(f.name() or hex(f.pc())); f = f.older()
                print("SAMPLE:" + ">".join(fr), flush=True)
        gdb.post_event(lambda: gdb.execute("continue"))
    elif isinstance(ev, gdb.ExitedEvent) or isinstance(ev, gdb.BreakpointEvent):
        pass
gdb.events.stop.connect(on_stop)
gdb.execute("handle SIGINT stop print nopass")
