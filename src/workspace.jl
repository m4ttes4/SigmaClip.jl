struct SigmaClipWorkspace{B <: AbstractVector, A <: Union{Nothing, AbstractVector}}
    buf::B
    aux::A
end

function prepare_ws(data::AbstractArray{T}, spread, ::Nothing) where {T}
    buf = Vector{T}(undef, length(data))
    aux = need_aux(spread) ? Vector{float(T)}(undef, length(data)) : nothing
    return SigmaClipWorkspace(buf, aux)
end

function prepare_ws(data::AbstractArray{T}, spread, ws::SigmaClipWorkspace) where {T}
    n = length(data)
    eltype(ws.buf) === T || throw(
        ArgumentError("workspace buffer element type $(eltype(ws.buf)) does not match $T")
    )
    length(ws.buf) >= n || throw(
        DimensionMismatch("workspace buffer has length $(length(ws.buf)); need at least $n")
    )

    if need_aux(spread)
        isnothing(ws.aux) && throw(ArgumentError("mad_std! requires an auxiliary buffer"))
        length(ws.aux) >= n || throw(
            DimensionMismatch("workspace auxiliary buffer has length $(length(ws.aux)); need at least $n")
        )
    end

    return ws
end
