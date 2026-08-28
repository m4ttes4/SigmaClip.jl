struct SigmaClipWorkspace{B <: AbstractVector, A <: Union{Nothing, AbstractVector}}
    buf::B
    aux::A
end

@inline workspace_buffer(ws::SigmaClipWorkspace) = ws.buf
@inline workspace_auxbuffer(ws::SigmaClipWorkspace) = ws.aux

"""
    workspace_buffer(ws) -> AbstractVector

Return the writable main scratch buffer used by SigmaClip. Custom workspace
types can implement this method to participate in the workspace API.
"""
workspace_buffer(ws) = throw(
    ArgumentError(
        "unsupported workspace $(typeof(ws)); implement SigmaClip.workspace_buffer and SigmaClip.workspace_auxbuffer"
    ),
)

"""
    workspace_auxbuffer(ws) -> AbstractVector or nothing

Return the auxiliary scratch buffer used by `mad_std!`, or `nothing` when it
is not available. Custom workspace types can implement this method to
participate in the workspace API.
"""
workspace_auxbuffer(ws) = throw(
    ArgumentError(
        "unsupported workspace $(typeof(ws)); implement SigmaClip.workspace_buffer and SigmaClip.workspace_auxbuffer"
    ),
)

function prepare_ws(data::AbstractArray{T}, spread, ::Nothing) where {T}
    buf = Vector{T}(undef, length(data))
    aux = need_aux(spread) ? Vector{float(T)}(undef, length(data)) : nothing
    return SigmaClipWorkspace(buf, aux)
end

function prepare_ws(data::AbstractArray{T}, spread, ws::WS) where {T, WS}
    n = length(data)
    buf = workspace_buffer(ws)
    aux = workspace_auxbuffer(ws)

    buf isa AbstractVector || throw(
        ArgumentError("workspace_buffer must return an AbstractVector")
    )
    eltype(buf) === T || throw(
        ArgumentError("workspace buffer element type $(eltype(buf)) does not match $T")
    )
    length(buf) >= n || throw(
        DimensionMismatch("workspace buffer has length $(length(buf)); need at least $n")
    )

    if need_aux(spread)
        isnothing(aux) && throw(ArgumentError("mad_std! requires an auxiliary buffer"))
        aux isa AbstractVector || throw(
            ArgumentError("workspace_auxbuffer must return an AbstractVector or nothing")
        )
        length(aux) >= n || throw(
            DimensionMismatch("workspace auxiliary buffer has length $(length(aux)); need at least $n")
        )
    end

    return ws
end
