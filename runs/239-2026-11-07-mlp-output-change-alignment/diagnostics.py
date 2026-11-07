"""Global cosine between consecutive prediction displacements."""
import torch


def output_alignment(outputs,states):
    values=outputs.double().flatten(1)
    changes=values[1:]-values[:-1]
    records=[]
    for i in range(1,len(changes)):
        previous,current=changes[i-1],changes[i]
        pn=float(previous.norm());cn=float(current.norm())
        records.append(dict(state=states[i],incoming_output_change_norm=pn,outgoing_output_change_norm=cn,
            output_change_cosine=float(previous@current)/(pn*cn) if pn*cn else None))
    return records
